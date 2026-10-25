using Application.Domain;
using System.Drawing;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Graphics.OpenGL;
using Microsoft.Extensions.Options;
using Application.RuntimeSettings;
using Application.Looper;

namespace Application.Visualization.Screen
{
    public class OpenTkScreenVisualizer : GameWindow, IVisualizer
    {
        private const string Font8X8Base64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGDw8GBgAGAA2NgAAAAAAADY2fzZ/NjYADD4DHjAfDAAAYzMYDGZjABw2HG47M24ABgYDAAAAAAAYDAYGBgwYAAYMGBgYDAYAAGY8/zxmAAAADAw/DAwAAAAAAAAADAwGAAAAPwAAAAAAAAAAAAwMAGAwGAwGAwEAPmNze29nPgAMDgwMDAw/AB4zMBwGMz8AHjMwHDAzHgA4PDYzfzB4AD8DHzAwMx4AHAYDHzMzHgA/MzAYDAwMAB4zMx4zMx4AHjMzPjAYDgAADAwAAAwMAAAMDAAADAwGGAwGAwYMGAAAAD8AAD8AAAYMGDAYDAYAHjMwGAwADAA+Y3t7ewMeAAweMzM/MzMAP2ZmPmZmPwA8ZgMDA2Y8AB82ZmZmNh8Af0YWHhZGfwB/RhYeFgYPADxmAwNzZnwAMzMzPzMzMwAeDAwMDAweAHgwMDAzMx4AZ2Y2HjZmZwAPBgYGRmZ/AGN3f39rY2MAY2dve3NjYwAcNmNjYzYcAD9mZj4GBg8AHjMzMzseOAA/ZmY+NmZnAB4zBw44Mx4APy0MDAwMHgAzMzMzMzM/ADMzMzMzHgwAY2Nja393YwBjYzYcHDZjADMzMx4MDB4Af2MxGExmfwAeBgYGBgYeAAMGDBgwYEAAHhgYGBgYHgAIHDZjAAAAAAAAAAAAAAD/DAwYAAAAAAAAAB4wPjNuAAcGBj5mZjsAAAAeMwMzHgA4MDA+MzNuAAAAHjM/Ax4AHDYGDwYGDwAAAG4zMz4wHwcGNm5mZmcADAAODAwMHgAwADAwMDMzHgcGZjYeNmcADgwMDAwMHgAAADN/f2tjAAAAHzMzMzMAAAAeMzMzHgAAADtmZj4GDwAAbjMzPjB4AAA7bmYGDwAAAD4DHjAfAAgMPgwMLBgAAAAzMzMzbgAAADMzMx4MAAAAY2t/fzYAAABjNhw2YwAAADMzMz4wHwAAPxkMJj8AOAwMBwwMOAAYGBgAGBgYAAcMDDgMDAcAbjsAAAAAAAAAAAAAAAAAAA==";

        private readonly IOptionsMonitor<StaticSettings> staticSettings;

        public OpenTkScreenVisualizer(GameWindowSettings gws, NativeWindowSettings nws, IOptionsMonitor<StaticSettings> staticSettings) : base(gws, nws)
        {
            this.staticSettings = staticSettings;
        }

        private volatile Color[] colors = Array.Empty<Color>();
        private volatile LooperPerformanceMetrics? latestMetrics;

        private int vao, vbo, instanceVbo;
        private int shaderProgram;
        private int instanceCount;

        private int modeLocation;
        private int lastMode = -1;
        private float[] instanceData = Array.Empty<float>();
        private int currentVboCapacity = 0;
        private float timePhase = 0f;

        // UI overlay rendering resources
        private int uiVao, uiVbo;
        private int uiShaderProgram;
        private int uiFontSamplerLocation;
        private int fontTexture;
        private float[] uiVertexData = Array.Empty<float>();
        private int uiVboCapacity = 0;

        public void UpdateColors(Color[] colors) => this.colors = colors;
        public void UpdateMetrics(LooperPerformanceMetrics metrics) => this.latestMetrics = metrics;

        public void Start() => Run();
        public void Stop() => Close();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                GL.DeleteProgram(shaderProgram);
                GL.DeleteProgram(uiShaderProgram);
                GL.DeleteBuffer(vbo);
                GL.DeleteBuffer(instanceVbo);
                GL.DeleteBuffer(uiVbo);
                GL.DeleteVertexArray(vao);
                GL.DeleteVertexArray(uiVao);
                GL.DeleteTexture(fontTexture);
            }

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

            InitFontTexture();
            CreateUiShaders();
            CreateUiBuffers();

            base.VSync = VSyncMode.On;
        }

        protected override void OnRenderFrame(FrameEventArgs args)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            var currentColors = this.colors;
            if (currentColors.Length > 0)
            {
                GL.UseProgram(shaderProgram);

                UpdateInstanceBuffer(currentColors);

                GL.BindVertexArray(vao);
                GL.DrawArraysInstanced(PrimitiveType.TriangleStrip, 0, 4, instanceCount);
            }

            if (this.staticSettings.CurrentValue.PrintFrameTimes && this.latestMetrics != null)
            {
                RenderPerformanceOverlay(this.latestMetrics);
            }

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

        private void InitFontTexture()
        {
            byte[] fontBytes = Convert.FromBase64String(Font8X8Base64);
            byte[] texData = new byte[128 * 64 * 4];

            for (int c = 0; c < 128; c++)
            {
                int charCol = c % 16;
                int charRow = c / 16;

                for (int py = 0; py < 8; py++)
                {
                    byte b = fontBytes[c * 8 + py];
                    for (int px = 0; px < 8; px++)
                    {
                        int bit = (b >> px) & 1;
                        int texX = charCol * 8 + px;
                        int texY = charRow * 8 + py;
                        int idx = (texY * 128 + texX) * 4;

                        texData[idx + 0] = 255;
                        texData[idx + 1] = 255;
                        texData[idx + 2] = 255;
                        texData[idx + 3] = bit != 0 ? (byte)255 : (byte)0;
                    }
                }
            }

            fontTexture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, fontTexture);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

            GL.TexImage2D(
                TextureTarget.Texture2D,
                0,
                PixelInternalFormat.Rgba,
                128,
                64,
                0,
                PixelFormat.Rgba,
                PixelType.UnsignedByte,
                texData
            );
        }

        private void CreateUiShaders()
        {
            string vs = @"
#version 130

in vec2 aPos;
in vec2 aTexCoord;
in vec4 aColor;

out vec2 vTexCoord;
out vec4 vColor;

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
    vColor = aColor;
}";

            string fs = @"
#version 130

in vec2 vTexCoord;
in vec4 vColor;

uniform sampler2D fontTexture;

out vec4 FragColor;

void main()
{
    if (vTexCoord.x < -0.5)
    {
        FragColor = vColor;
    }
    else
    {
        float alpha = texture(fontTexture, vTexCoord).a;
        FragColor = vec4(vColor.rgb, vColor.a * alpha);
    }
}";

            int v = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(v, vs);
            GL.CompileShader(v);

            int f = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(f, fs);
            GL.CompileShader(f);

            uiShaderProgram = GL.CreateProgram();
            GL.AttachShader(uiShaderProgram, v);
            GL.AttachShader(uiShaderProgram, f);

            GL.BindAttribLocation(uiShaderProgram, 0, "aPos");
            GL.BindAttribLocation(uiShaderProgram, 1, "aTexCoord");
            GL.BindAttribLocation(uiShaderProgram, 2, "aColor");

            GL.LinkProgram(uiShaderProgram);

            uiFontSamplerLocation = GL.GetUniformLocation(uiShaderProgram, "fontTexture");

            GL.DeleteShader(v);
            GL.DeleteShader(f);
        }

        private void CreateUiBuffers()
        {
            uiVao = GL.GenVertexArray();
            uiVbo = GL.GenBuffer();

            GL.BindVertexArray(uiVao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, uiVbo);

            int stride = 8 * sizeof(float);

            // aPos
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, stride, 0);

            // aTexCoord
            GL.EnableVertexAttribArray(1);
            GL.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, stride, 2 * sizeof(float));

            // aColor
            GL.EnableVertexAttribArray(2);
            GL.VertexAttribPointer(2, 4, VertexAttribPointerType.Float, false, stride, 4 * sizeof(float));
        }

        private void RenderPerformanceOverlay(LooperPerformanceMetrics metrics)
        {
            float screenW = FramebufferSize.X;
            float screenH = FramebufferSize.Y;
            if (screenW <= 0 || screenH <= 0)
                return;

            string sFps = $"FPS: {(int)Math.Round(metrics.ActualFps)}";
            string sFrame = $"Frame: {metrics.FrameTimeMs:F2} ms";
            string sProcess = $"Process: {metrics.ProcessTimeMs:F2} ms";
            string sTheo = $"Theo FPS: {(double.IsInfinity(metrics.TheoreticalFps) ? "inf" : metrics.TheoreticalFps.ToString("F1"))}";

            string[] badges = new[] { sFps, sFrame, sProcess, sTheo };

            float scale = (screenW >= 1200f && screenH >= 600f) ? 2f : 1f;
            float charW = 8f * scale;
            float charH = 8f * scale;

            float padLeft = 10f;
            float padTop = 10f;
            float boxPadX = 6f * scale;
            float boxPadY = 4f * scale;
            float boxSpacing = 6f * scale;

            float boxH = charH + 2f * boxPadY;
            float currentX = padLeft;

            int totalQuads = badges.Length;
            foreach (var b in badges)
                totalQuads += b.Length;

            int requiredFloats = totalQuads * 6 * 8;
            if (uiVertexData.Length < requiredFloats)
            {
                uiVertexData = new float[Math.Max(requiredFloats, 2048)];
            }

            int vIdx = 0;

            // Box background: dark semi-transparent
            float bgR = 15f / 255f, bgG = 18f / 255f, bgB = 25f / 255f, bgA = 0.80f;
            // Text color: crisp light gray / white
            float textR = 240f / 255f, textG = 240f / 255f, textB = 245f / 255f, textA = 0.95f;

            for (int b = 0; b < badges.Length; b++)
            {
                string text = badges[b];
                float textW = text.Length * charW;
                float boxW = textW + 2f * boxPadX;

                float bx0 = currentX;
                float by0 = padTop;
                float bx1 = bx0 + boxW;
                float by1 = by0 + boxH;

                // Add background quad (u = -1, v = -1 indicates solid color)
                AddQuad(ref vIdx, screenW, screenH, bx0, by0, bx1, by1, -1f, -1f, -1f, -1f, bgR, bgG, bgB, bgA);

                // Add character quads
                float charStartX = bx0 + boxPadX;
                float charStartY = by0 + boxPadY;

                for (int i = 0; i < text.Length; i++)
                {
                    char c = text[i];
                    int charIdx = (int)c;
                    if (charIdx < 0 || charIdx >= 128)
                        charIdx = '?';

                    int charCol = charIdx % 16;
                    int charRow = charIdx / 16;

                    float u0 = (charCol * 8f) / 128f;
                    float u1 = ((charCol + 1) * 8f) / 128f;
                    float v0 = (charRow * 8f) / 64f;
                    float v1 = ((charRow + 1) * 8f) / 64f;

                    float cx0 = charStartX + i * charW;
                    float cy0 = charStartY;
                    float cx1 = cx0 + charW;
                    float cy1 = cy0 + charH;

                    AddQuad(ref vIdx, screenW, screenH, cx0, cy0, cx1, cy1, u0, v0, u1, v1, textR, textG, textB, textA);
                }

                currentX += boxW + boxSpacing;
            }

            // Upload to VBO
            GL.BindBuffer(BufferTarget.ArrayBuffer, uiVbo);
            int byteCount = vIdx * sizeof(float);
            if (byteCount > uiVboCapacity)
            {
                GL.BufferData(BufferTarget.ArrayBuffer, byteCount, uiVertexData, BufferUsageHint.StreamDraw);
                uiVboCapacity = byteCount;
            }
            else
            {
                GL.BufferSubData(BufferTarget.ArrayBuffer, IntPtr.Zero, byteCount, uiVertexData);
            }

            // Draw UI
            GL.UseProgram(uiShaderProgram);
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, fontTexture);
            GL.Uniform1(uiFontSamplerLocation, 0);

            GL.BindVertexArray(uiVao);
            GL.DrawArrays(PrimitiveType.Triangles, 0, vIdx / 8);
        }

        private void AddQuad(
            ref int vIdx,
            float screenW, float screenH,
            float x0, float y0, float x1, float y1,
            float u0, float v0, float u1, float v1,
            float r, float g, float b, float a)
        {
            float ndcX0 = (x0 / screenW) * 2f - 1f;
            float ndcX1 = (x1 / screenW) * 2f - 1f;
            float ndcY0 = 1f - (y0 / screenH) * 2f;
            float ndcY1 = 1f - (y1 / screenH) * 2f;

            // Triangle 1: (x0, y0), (x1, y0), (x0, y1)
            WriteVertex(ref vIdx, ndcX0, ndcY0, u0, v0, r, g, b, a);
            WriteVertex(ref vIdx, ndcX1, ndcY0, u1, v0, r, g, b, a);
            WriteVertex(ref vIdx, ndcX0, ndcY1, u0, v1, r, g, b, a);

            // Triangle 2: (x1, y0), (x1, y1), (x0, y1)
            WriteVertex(ref vIdx, ndcX1, ndcY0, u1, v0, r, g, b, a);
            WriteVertex(ref vIdx, ndcX1, ndcY1, u1, v1, r, g, b, a);
            WriteVertex(ref vIdx, ndcX0, ndcY1, u0, v1, r, g, b, a);
        }

        private void WriteVertex(
            ref int vIdx,
            float x, float y,
            float u, float v,
            float r, float g, float b, float a)
        {
            uiVertexData[vIdx + 0] = x;
            uiVertexData[vIdx + 1] = y;
            uiVertexData[vIdx + 2] = u;
            uiVertexData[vIdx + 3] = v;
            uiVertexData[vIdx + 4] = r;
            uiVertexData[vIdx + 5] = g;
            uiVertexData[vIdx + 6] = b;
            uiVertexData[vIdx + 7] = a;
            vIdx += 8;
        }
    }
}
