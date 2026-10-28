// Turn a large white-on-black drawing into a 28x28 image that looks like MNIST.
//
// MNIST digits were prepared like this:
//   1. The digit was scaled to fit in a 20x20 box, keeping its aspect ratio.
//   2. It was placed in a 28x28 black image so its center of mass is in the middle.
// The steps below follow the same recipe so the drawing matches what the CNN learned on.

export const MNIST_SIZE = 28;
const DIGIT_BOX_SIZE = 20;
const INK_THRESHOLD = 30; // pixel values (0-255) above this count as part of the digit

function createCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function readGray(canvas) {
  const { width, height } = canvas;
  const rgba = canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, width, height).data;
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < gray.length; i++) {
    // The drawing is white on black, so the red channel already holds the gray value.
    gray[i] = rgba[i * 4];
  }
  return gray;
}

function findInkBounds(gray, width, height) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (gray[y * width + x] > INK_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

/** Crop a region and shrink it in halving steps, which looks much smoother than one big jump. */
function resizeRegion(source, region, targetWidth, targetHeight) {
  let current = createCanvas(region.width, region.height);
  current.getContext("2d").drawImage(
    source, region.x, region.y, region.width, region.height, 0, 0, region.width, region.height,
  );

  while (current.width / 2 >= targetWidth && current.height / 2 >= targetHeight) {
    const half = createCanvas(Math.floor(current.width / 2), Math.floor(current.height / 2));
    const ctx = half.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(current, 0, 0, half.width, half.height);
    current = half;
  }

  const result = createCanvas(targetWidth, targetHeight);
  const ctx = result.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(current, 0, 0, targetWidth, targetHeight);
  return result;
}

function centerOfMass(gray, width, height) {
  let total = 0;
  let sumX = 0;
  let sumY = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = gray[y * width + x];
      total += value;
      sumX += x * value;
      sumY += y * value;
    }
  }
  if (total === 0) return { x: (width - 1) / 2, y: (height - 1) / 2 };
  return { x: sumX / total, y: sumY / total };
}

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

/**
 * Convert the drawing canvas to 28x28 grayscale pixels (Uint8Array, row by row,
 * 0 = black background, 255 = white ink). Returns null if nothing has been drawn.
 */
export function canvasToMnistPixels(sourceCanvas) {
  const sourceGray = readGray(sourceCanvas);
  const bounds = findInkBounds(sourceGray, sourceCanvas.width, sourceCanvas.height);
  if (!bounds) return null;

  // 1. Fit the digit into a 20x20 box without stretching it.
  const scale = DIGIT_BOX_SIZE / Math.max(bounds.width, bounds.height);
  const digitWidth = Math.max(1, Math.round(bounds.width * scale));
  const digitHeight = Math.max(1, Math.round(bounds.height * scale));
  const digit = resizeRegion(sourceCanvas, bounds, digitWidth, digitHeight);

  // 2. Shift it so its center of mass lands in the middle of the 28x28 image.
  const mass = centerOfMass(readGray(digit), digitWidth, digitHeight);
  const offsetX = clamp(Math.round(MNIST_SIZE / 2 - mass.x), 0, MNIST_SIZE - digitWidth);
  const offsetY = clamp(Math.round(MNIST_SIZE / 2 - mass.y), 0, MNIST_SIZE - digitHeight);

  const output = createCanvas(MNIST_SIZE, MNIST_SIZE);
  const ctx = output.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, MNIST_SIZE, MNIST_SIZE);
  ctx.drawImage(digit, offsetX, offsetY);

  return readGray(output);
}

// ----- Minimal 8-bit grayscale PNG encoder -----
// Browsers can only export color (RGBA) PNGs from a canvas, so we build a true
// single-channel PNG by hand. CompressionStream("deflate") produces exactly the
// zlib data a PNG needs.

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const chunk = new Uint8Array(12 + data.length);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) chunk[4 + i] = type.charCodeAt(i);
  chunk.set(data, 8);
  view.setUint32(8 + data.length, crc32(chunk.subarray(4, 8 + data.length)));
  return chunk;
}

async function zlibCompress(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function encodeGrayscalePng(pixels, width, height) {
  const header = new Uint8Array(13);
  const view = new DataView(header.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  header[8] = 8; // bits per pixel
  header[9] = 0; // color type 0 = grayscale
  // bytes 10-12: compression, filter, and interlace methods are all 0

  // Each row starts with a filter-type byte (0 = no filter).
  const raw = new Uint8Array((width + 1) * height);
  for (let y = 0; y < height; y++) {
    raw.set(pixels.subarray(y * width, (y + 1) * width), y * (width + 1) + 1);
  }

  const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  return new Blob(
    [
      signature,
      pngChunk("IHDR", header),
      pngChunk("IDAT", await zlibCompress(raw)),
      pngChunk("IEND", new Uint8Array(0)),
    ],
    { type: "image/png" },
  );
}

/** Fallback for browsers without CompressionStream: a regular canvas PNG (gray stored as RGB). */
function encodeWithCanvas(pixels, width, height) {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(width, height);
  pixels.forEach((value, i) => {
    image.data.set([value, value, value, 255], i * 4);
  });
  ctx.putImageData(image, 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not create PNG."))), "image/png");
  });
}

/**
 * Preprocess the drawing and encode it as a 28x28 grayscale PNG.
 * Resolves to a Blob, or null if the canvas is empty.
 */
export async function canvasToMnistPng(sourceCanvas) {
  const pixels = canvasToMnistPixels(sourceCanvas);
  if (!pixels) return null;
  if (typeof CompressionStream === "function") {
    return encodeGrayscalePng(pixels, MNIST_SIZE, MNIST_SIZE);
  }
  return encodeWithCanvas(pixels, MNIST_SIZE, MNIST_SIZE);
}
