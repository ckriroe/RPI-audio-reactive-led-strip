using Application.Domain;
using System.Drawing;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Graphics.OpenGL;
using Microsoft.Extensions.Options;
using Application.RuntimeSettings;

namespace Application.Visualization.Screen
{
    public class OpenTkScreenVisualizer : GameWindow, IVisualizer
    {
        private readonly IOptionsMonitor<StaticSettings> staticSettings;

        public OpenTkScreenVisualizer(GameWindowSettings gws, NativeWindowSettings nws, IOptionsMonitor<StaticSettings> staticSettings) : base(gws, nws)
        {
            this.staticSettings = staticSettings;
        }

        private volatile Color[] colors = Array.Empty<Color>();

        private int vao, vbo, instanceVbo;
        private int shaderProgram;
        private int instanceCount;

        private int modeLocation;
        private int lastMode = -1;
        private float[] instanceData = Array.Empty<float>();
        private int currentVboCapacity = 0;
        private float timePhase = 0f;

        public void UpdateColors(Color[] colors) => this.colors = colors;

        public void Start() => Run();
        public void Stop() => Close();

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
        }

        protected override void OnLoad()
        {
            GL.ClearColor(Color.Black);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

            CreateShaders();
            CreateQuad();
            CreateInstanceBuffer();
            base.VSync = VSyncMode.On;
        }

        protected override void OnRenderFrame(FrameEventArgs args)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            var currentColors = this.colors;
            if (currentColors.Length == 0)
            {
                SwapBuffers();
                return;
            }

            GL.UseProgram(shaderProgram);

            UpdateInstanceBuffer(currentColors);

            GL.BindVertexArray(vao);
            GL.DrawArraysInstanced(PrimitiveType.TriangleStrip, 0, 4, instanceCount);

            SwapBuffers();
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(0, 0, FramebufferSize.X, FramebufferSize.Y);
        }

        private void CreateQuad()
        {
            float[] vertices =
            {
                -1f, -1f,
                 1f, -1f,
                -1f,  1f,
                 1f,  1f
            };

            vao = GL.GenVertexArray();
            vbo = GL.GenBuffer();

            GL.BindVertexArray(vao);

            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
            GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.StaticDraw);

            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, 2 * sizeof(float), 0);
        }

        private void CreateInstanceBuffer()
        {
            instanceVbo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, instanceVbo);

            int stride = 8 * sizeof(float);

            // pos
            GL.EnableVertexAttribArray(1);
            GL.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, stride, 0);
            GL.VertexAttribDivisor(1, 1);

            // scale
            GL.EnableVertexAttribArray(2);
            GL.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, 2 * sizeof(float));
            GL.VertexAttribDivisor(2, 1);

            // color
            GL.EnableVertexAttribArray(3);
            GL.VertexAttribPointer(3, 3, VertexAttribPointerType.Float, false, stride, 4 * sizeof(float));
            GL.VertexAttribDivisor(3, 1);

            // extra (inner radius for rings)
            GL.EnableVertexAttribArray(4);
            GL.VertexAttribPointer(4, 1, VertexAttribPointerType.Float, false, stride, 7 * sizeof(float));
            GL.VertexAttribDivisor(4, 1);
        }

        private void EnsureDataCapacity(int count)
        {
            int requiredLength = count * 8;
            if (instanceData.Length < requiredLength)
            {
                instanceData = new float[Math.Max(instanceData.Length * 2, Math.Max(requiredLength, 1024))];
            }
        }

        private void Upload(int count)
        {
            GL.BindBuffer(BufferTarget.ArrayBuffer, instanceVbo);
            int requiredBytes = count * 8 * sizeof(float);

            if (requiredBytes > currentVboCapacity)
            {
                GL.BufferData(BufferTarget.ArrayBuffer, requiredBytes, instanceData, BufferUsageHint.StreamDraw);
                currentVboCapacity = requiredBytes;
            }
            else
            {
                GL.BufferSubData(BufferTarget.ArrayBuffer, IntPtr.Zero, requiredBytes, instanceData);
            }
        }

        private void UpdateInstanceBuffer(Color[] currentColors)
        {
            var mode = this.staticSettings.CurrentValue.GuiVisualizationMode;

            switch (mode)
            {
                case GuiVisualizationMode.Dots:
                    UpdateDots(currentColors);
                    SetModeUniform(0);
                    break;

                case GuiVisualizationMode.Rectangles:
                    UpdateRectangles(currentColors);
                    SetModeUniform(1);
                    break;

                case GuiVisualizationMode.RadialFromFirst:
                    UpdateRadial(currentColors);
                    SetModeUniform(2);
                    break;

                case GuiVisualizationMode.Particle3D:
                    UpdateParticles3D(currentColors);
                    SetModeUniform(3);
                    break;
            }
        }

        private void SetModeUniform(int modeValue)
        {
            if (lastMode != modeValue)
            {
                GL.Uniform1(modeLocation, modeValue);
                lastMode = modeValue;
            }
        }

        private void UpdateDots(Color[] currentColors)
        {
            int count = currentColors.Length;
            EnsureDataCapacity(count);

            float width = FramebufferSize.X;
            float height = FramebufferSize.Y;
            float spacing = width / count;

            float spacingX = spacing / width;
            float spacingY = spacing / height;
            const float colorFactor = 1f / 255f;

            for (int i = 0; i < count; i++)
            {
                float x = ((i + 0.5f) * spacing) / width * 2f - 1f;
                int idx = i * 8;
                var c = currentColors[i];

                instanceData[idx + 0] = x;
                instanceData[idx + 1] = 0f;
                instanceData[idx + 2] = spacingX;
                instanceData[idx + 3] = spacingY;
                instanceData[idx + 4] = c.R * colorFactor;
                instanceData[idx + 5] = c.G * colorFactor;
                instanceData[idx + 6] = c.B * colorFactor;
                instanceData[idx + 7] = 0f;
            }

            Upload(count);
            instanceCount = count;
        }

        private void UpdateRectangles(Color[] currentColors)
        {
            int count = currentColors.Length;
            EnsureDataCapacity(count);

            int rectangleHeightPx = this.staticSettings.CurrentValue.RectangleGuiVisualizationHeight;

            float width = FramebufferSize.X;
            float height = FramebufferSize.Y;
            float spacing = width / count;

            float spacingX = spacing / width;
            float scaleY = rectangleHeightPx / height;
            const float colorFactor = 1f / 255f;

            for (int i = 0; i < count; i++)
            {
                float x = ((i + 0.5f) * spacing) / width * 2f - 1f;
                int idx = i * 8;
                var c = currentColors[i];

                instanceData[idx + 0] = x;
                instanceData[idx + 1] = 0f;
                instanceData[idx + 2] = spacingX;
                instanceData[idx + 3] = scaleY;
                instanceData[idx + 4] = c.R * colorFactor;
                instanceData[idx + 5] = c.G * colorFactor;
                instanceData[idx + 6] = c.B * colorFactor;
                instanceData[idx + 7] = 0f;
            }

            Upload(count);
            instanceCount = count;
        }

        private void UpdateRadial(Color[] currentColors)
        {
            int count = currentColors.Length;
            EnsureDataCapacity(count);

            float step = 1f / count;
            const float colorFactor = 1f / 255f;

            for (int i = 0; i < count; i++)
            {
                float outer = step * (i + 1);
                float inner = step * i;
                int idx = i * 8;
                var c = currentColors[i];

                instanceData[idx + 0] = 0f;
                instanceData[idx + 1] = 0f;
                instanceData[idx + 2] = outer;
                instanceData[idx + 3] = outer;
                instanceData[idx + 4] = c.R * colorFactor;
                instanceData[idx + 5] = c.G * colorFactor;
                instanceData[idx + 6] = c.B * colorFactor;
                instanceData[idx + 7] = inner;
            }

            Upload(count);
            instanceCount = count;
        }

        private void UpdateParticles3D(Color[] currentColors)
        {
            int count = currentColors.Length;
            EnsureDataCapacity(count);

            float aspect = (float)FramebufferSize.X / FramebufferSize.Y;
            const float colorFactor = 1f / 255f;

            timePhase += 0.015f;

            float phi = MathF.PI * (3f - MathF.Sqrt(5f));


            for (int i = 0; i < count; i++)
            {
                // 1. Map to a Fibonacci Sphere
                float y = 1f - (i / (float)(count - 1)) * 2f; // y goes from 1 to -1
                float radiusAtY = MathF.Sqrt(1f - y * y);     // radius at y
                float theta = phi * i;                        // golden angle increment

                float x3d = MathF.Cos(theta) * radiusAtY;
                float z3d = MathF.Sin(theta) * radiusAtY;

                // 2. Apply Organic "Breathing" and Morphing
                // Creates complex interference waves based on position and time
                float wave1 = MathF.Sin(timePhase * 2.0f + y * 5.0f);
                float wave2 = MathF.Cos(timePhase * 1.5f + theta * 3.0f);
                float pulse = 0.6f + 0.5f * (wave1 * wave2);

                x3d *= pulse;
                y *= pulse;
                z3d *= pulse;

                // 3. 3D Rotation (Tumble the shape)
                float rotX = timePhase * 0.4f;
                float cosX = MathF.Cos(rotX), sinX = MathF.Sin(rotX);
                float y2 = y * cosX - z3d * sinX;
                float z2 = y * sinX + z3d * cosX;

                float rotY = timePhase * 0.7f;
                float cosY = MathF.Cos(rotY), sinY = MathF.Sin(rotY);
                float x3 = x3d * cosY - z2 * sinY;
                float z3 = x3d * sinY + z2 * cosY;

                // 4. Perspective Projection
                float zCam = z3 + 1.75f; // Push away from camera
                float projX = x3 / zCam / aspect;
                float projY = y2 / zCam;

                // 5. Calculate Scale & Depth Fog
                float scale = 0.12f / zCam;

                // Darken particles that are further away in the Z-axis
                float depthFog = Math.Clamp(1.5f - zCam * 0.45f, 0.05f, 1.0f);

                int idx = i * 8;
                var c = currentColors[i];

                instanceData[idx + 0] = projX;
                instanceData[idx + 1] = projY;
                instanceData[idx + 2] = scale;
                instanceData[idx + 3] = scale * aspect;

                instanceData[idx + 4] = (c.R * colorFactor) * depthFog;
                instanceData[idx + 5] = (c.G * colorFactor) * depthFog;
                instanceData[idx + 6] = (c.B * colorFactor) * depthFog;
                instanceData[idx + 7] = 0f;
            }

            Upload(count);
            instanceCount = count;
        }

        private void CreateShaders()
        {
            string vs = @"
#version 130

in vec2 aPos;
in vec2 instancePos;
in vec2 instanceScale;
in vec3 instanceColor;
in float innerRadius;

out vec2 vLocal;
out vec3 vColor;
out float vInner;

void main()
{
    vec2 pos = aPos * instanceScale + instancePos;
    gl_Position = vec4(pos, 0.0, 1.0);

    vLocal = aPos;
    vColor = instanceColor;
    vInner = innerRadius;
}";

            string fs = @"
#version 130

in vec2 vLocal;
in vec3 vColor;
in float vInner;

uniform int mode;

out vec4 FragColor;

void main()
{
    float dist = length(vLocal);

    if (mode == 0) // dots
    {
        if (dist > 1.0) discard;
    }
    else if (mode == 2) // rings
    {
        if (dist > 1.0 || dist < vInner) discard;
    }
    else if (mode == 3) // 3D Sharp Particles
    {
        if (dist > 1.0) discard;
        
        float alpha = pow(1.0 - dist, 5.0); 
        
        if (dist < 0.15) alpha = 1.0;
        
        // Calculate the brightness of the particle (0.0 for black, 1.0 for bright colors)
        float brightness = max(max(vColor.r, vColor.g), vColor.b);
        
        // Tie the opacity to the brightness so black becomes totally transparent
        alpha *= brightness;
        
        // Optimization: if it is practically invisible, discard it early to save GPU cycles
        if (alpha < 0.01) discard;
        
        FragColor = vec4(vColor, alpha);
        return; 
    }

    // Default color for rectangles, dots, and rings
    FragColor = vec4(vColor, 1.0);
}";

            int v = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(v, vs);
            GL.CompileShader(v);

            int f = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(f, fs);
            GL.CompileShader(f);

            shaderProgram = GL.CreateProgram();
            GL.AttachShader(shaderProgram, v);
            GL.AttachShader(shaderProgram, f);

            GL.BindAttribLocation(shaderProgram, 0, "aPos");
            GL.BindAttribLocation(shaderProgram, 1, "instancePos");
            GL.BindAttribLocation(shaderProgram, 2, "instanceScale");
            GL.BindAttribLocation(shaderProgram, 3, "instanceColor");
            GL.BindAttribLocation(shaderProgram, 4, "innerRadius");

            GL.LinkProgram(shaderProgram);

            modeLocation = GL.GetUniformLocation(shaderProgram, "mode");

            GL.DeleteShader(v);
            GL.DeleteShader(f);
        }
    }
}
