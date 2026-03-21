using Application.Domain;
using OpenTK.Graphics.OpenGL;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using System.Drawing;

namespace Application.Visualization.Screen
{
    public class ScreenVisualizer : GameWindow
    {
        public ScreenVisualizer(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings) : base(gameWindowSettings, nativeWindowSettings)
        {
        }

        private volatile Color[] colors = [];

        public void UpdateColors(Color[] colors)
        {
            this.colors = colors;
        }

        protected override void OnLoad()
        {
            base.OnLoad();

            GL.ClearColor(Color.Black);
            GL.PointSize(6f);
            VSync = VSyncMode.On;
        }

        protected override void OnRenderFrame(FrameEventArgs args)
        {
            var colors = this.colors;
            GL.Clear(ClearBufferMask.ColorBufferBit);

            int count = colors.Length;
            if (count == 0)
                return;

            float width = Size.X;
            float height = Size.Y;
            float spacing = width / count;

            // radius in pixels
            float radiusPx = 6f;

            // convert radius to NDC (-1 to 1)
            float radiusNdcX = radiusPx / width * 2f;
            float radiusNdcY = radiusPx / height * 2f;

            for (int i = 0; i < count; i++)
            {
                var c = colors[i];

                // normalized device coordinates
                float xNdc = ((i + 0.5f) * spacing) / width * 2f - 1f;
                float yNdc = 0f; // center vertically

                DrawCircle(xNdc, yNdc, radiusNdcX, radiusNdcY, c);
            }

            SwapBuffers();
        }

        /// <summary>
        /// Draws a circle using a triangle fan.
        /// radiusX/Y are in NDC coordinates (-1..1)
        /// </summary>
        private void DrawCircle(float x, float y, float radiusX, float radiusY, Color c, int segments = 20)
        {
            GL.Color3(c.R / 255f, c.G / 255f, c.B / 255f);
            GL.Begin(PrimitiveType.TriangleFan);

            // center
            GL.Vertex2(x, y);

            for (int i = 0; i <= segments; i++)
            {
                double angle = i * 2.0 * Math.PI / segments;
                double dx = Math.Cos(angle) * radiusX;
                double dy = Math.Sin(angle) * radiusY;
                GL.Vertex2(x + dx, y + dy);
            }

            GL.End();
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);

            // Update the OpenGL viewport
            GL.Viewport(0, 0, Size.X, Size.Y);

            // Force redraw
            //this.Invalidate(); // requests OnRenderFrame to run again
        }
    }
}
