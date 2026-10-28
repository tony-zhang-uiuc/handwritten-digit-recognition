import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";

// The canvas always has this internal resolution; CSS scales it to fit the screen.
const CANVAS_SIZE = 560;
// About as thick, relative to the digit, as MNIST pen strokes.
const BRUSH_WIDTH = CANVAS_SIZE * 0.07;

function toCanvasPoint(canvas, rect, event) {
  return {
    x: ((event.clientX - rect.left) * canvas.width) / rect.width,
    y: ((event.clientY - rect.top) * canvas.height) / rect.height,
  };
}

/**
 * Black square canvas for drawing a white digit with a mouse, finger, or stylus.
 * The parent gets `clear()` and `getCanvas()` through the ref.
 */
const DrawingCanvas = forwardRef(function DrawingCanvas({ onDraw }, ref) {
  const canvasRef = useRef(null);
  // The stroke in progress: which pointer is drawing, plus the last point and midpoint.
  const strokeRef = useRef(null);

  const clear = useCallback(() => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  useEffect(clear, [clear]);

  useImperativeHandle(ref, () => ({ clear, getCanvas: () => canvasRef.current }), [clear]);

  function getBrush() {
    const ctx = canvasRef.current.getContext("2d");
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = BRUSH_WIDTH;
    ctx.strokeStyle = "#fff";
    ctx.fillStyle = "#fff";
    return ctx;
  }

  function handlePointerDown(event) {
    if (strokeRef.current) return; // ignore a second finger while one is drawing
    if (event.pointerType === "mouse" && event.button !== 0) return;
    event.preventDefault();

    const canvas = canvasRef.current;
    canvas.setPointerCapture(event.pointerId);
    const point = toCanvasPoint(canvas, canvas.getBoundingClientRect(), event);

    // A single tap should still leave a round dot.
    const ctx = getBrush();
    ctx.beginPath();
    ctx.arc(point.x, point.y, BRUSH_WIDTH / 2, 0, Math.PI * 2);
    ctx.fill();

    strokeRef.current = { pointerId: event.pointerId, last: point, lastMid: point };
    onDraw?.();
  }

  function handlePointerMove(event) {
    const stroke = strokeRef.current;
    if (!stroke || event.pointerId !== stroke.pointerId) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const ctx = getBrush();

    // Coalesced events include the in-between points the browser batched together,
    // which keeps fast strokes smooth instead of jagged.
    const coalesced = event.nativeEvent.getCoalescedEvents?.();
    const events = coalesced?.length ? coalesced : [event.nativeEvent];

    for (const pointerEvent of events) {
      const point = toCanvasPoint(canvas, rect, pointerEvent);
      const mid = { x: (stroke.last.x + point.x) / 2, y: (stroke.last.y + point.y) / 2 };

      // Curve from the previous midpoint to this one, bending through the last real point.
      ctx.beginPath();
      ctx.moveTo(stroke.lastMid.x, stroke.lastMid.y);
      ctx.quadraticCurveTo(stroke.last.x, stroke.last.y, mid.x, mid.y);
      ctx.stroke();

      stroke.last = point;
      stroke.lastMid = mid;
    }
  }

  function handlePointerUp(event) {
    const stroke = strokeRef.current;
    if (!stroke || event.pointerId !== stroke.pointerId) return;

    const ctx = getBrush();
    ctx.beginPath();
    ctx.moveTo(stroke.lastMid.x, stroke.lastMid.y);
    ctx.lineTo(stroke.last.x, stroke.last.y);
    ctx.stroke();

    strokeRef.current = null;
  }

  return (
    <canvas
      ref={canvasRef}
      className="drawing-canvas"
      width={CANVAS_SIZE}
      height={CANVAS_SIZE}
      role="img"
      aria-label="Drawing area. Draw a single digit from 0 to 9 with your mouse, finger, or stylus."
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onContextMenu={(event) => event.preventDefault()}
    />
  );
});

export default DrawingCanvas;
