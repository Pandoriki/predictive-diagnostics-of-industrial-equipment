import React, { useRef, useEffect } from 'react';
import { Theme } from '../types';

interface WebGLBackgroundProps {
  theme: Theme;
}

const vertexShaderSource = `
  attribute vec2 a_position;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

const fragmentShaderSource = `
  precision mediump float;
  uniform vec2 u_resolution;
  uniform float u_time;
  uniform vec2 u_mouse;
  uniform vec3 u_color1;
  uniform vec3 u_color2;
  uniform vec3 u_color3;

  float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
  }

  // Based on Morgan McGuire @morgan3d
  // https://www.shadertoy.com/view/4dS3Wd
  float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
  }

  // Optimization: Reduced octaves from 5 to 4 for better performance.
  #define NUM_OCTAVES 4

  float fbm ( in vec2 _st) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    // Rotate to reduce axial bias
    mat2 rot = mat2(cos(0.5), sin(0.5),
                    -sin(0.5), cos(0.50));
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(_st);
        _st = rot * _st * 2.0 + shift;
        a *= 0.5;
    }
    return v;
  }

  void main() {
    vec2 st = gl_FragCoord.xy/u_resolution.xy;
    st.x *= u_resolution.x/u_resolution.y;

    vec3 color = vec3(0.0);
    
    // Add mouse influence to the coordinate system
    vec2 mouse_influence = (u_mouse - 0.5) * 0.2;

    vec2 q = vec2(0.);
    q.x = fbm( st + 0.00*u_time + mouse_influence);
    q.y = fbm( st + vec2(1.0));

    vec2 r = vec2(0.);
    r.x = fbm( st + 1.0*q + vec2(1.7,9.2)+ 0.15*u_time );
    r.y = fbm( st + 1.0*q + vec2(8.3,2.8)+ 0.126*u_time);

    float f = fbm(st+r);

    color = mix(u_color1,
                u_color2,
                clamp((f*f)*4.0,0.0,1.0));

    color = mix(color,
                u_color3,
                clamp(length(q),0.0,1.0));

    color = mix(color,
                u_color1,
                clamp(length(r.xy),0.0,1.0));
    
    vec3 final_color_intensity = (f*f*f+.6*f*f+.5*f)*color;

    // Add vignette effect to darken edges and draw focus to the center
    vec2 uv_vignette = gl_FragCoord.xy/u_resolution.xy;
    float vignette = 1.0 - smoothstep(0.5, 1.0, length(uv_vignette - 0.5));
    final_color_intensity *= vignette;

    gl_FragColor = vec4(final_color_intensity, 1.0);
  }
`;

const WebGLBackground: React.FC<WebGLBackgroundProps> = ({ theme }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const animationFrameIdRef = useRef<number>(0);
  const buffersRef = useRef<{ position: WebGLBuffer | null }>({ position: null });

  const mousePosRef = useRef({ current: { x: 0.5, y: 0.5 }, target: { x: 0.5, y: 0.5 } });

  const colorsRef = useRef({
    color1: [0.01, 0.02, 0.08],
    color2: [0.05, 0.1, 0.3],
    color3: [0.1, 0.3, 0.6],
  });

  const transitionRef = useRef<{
    startTime: number | null,
    from: typeof colorsRef.current,
    to: typeof colorsRef.current,
    duration: number,
  }>({ startTime: null, from: colorsRef.current, to: colorsRef.current, duration: 500 });


  const themeColors = {
    [Theme.Dark]: {
      color1: [0.01, 0.02, 0.08], // Deep Blue
      color2: [0.05, 0.1, 0.3],   // Mid Blue
      color3: [0.1, 0.3, 0.6],    // Cyan/Light Blue
    },
    [Theme.Light]: {
      color1: [0.9, 0.95, 1.0], // Very light blue
      color2: [0.7, 0.8, 1.0],   // Soft blue
      color3: [1.0, 0.8, 0.9],   // Soft pink
    }
  };

  useEffect(() => {
    transitionRef.current = {
      startTime: Date.now(),
      from: { ...colorsRef.current },
      to: themeColors[theme],
      duration: 500,
    };
  }, [theme]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl', { antialias: true, powerPreference: 'low-power' });
    if (!gl) {
      console.error("WebGL not supported");
      return;
    }
    glRef.current = gl;

    const vertexShader = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);

    const program = gl.createProgram()!;
    programRef.current = program;

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Unable to initialize the shader program: ' + gl.getProgramInfoLog(program));
        return;
    }

    gl.useProgram(program);

    buffersRef.current.position = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.position);
    const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    const positionAttributeLocation = gl.getAttribLocation(program, "a_position");
    const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
    const timeLocation = gl.getUniformLocation(program, "u_time");
    const mouseLocation = gl.getUniformLocation(program, "u_mouse");
    const color1Location = gl.getUniformLocation(program, "u_color1");
    const color2Location = gl.getUniformLocation(program, "u_color2");
    const color3Location = gl.getUniformLocation(program, "u_color3");
    
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    };
    
    const handleMouseMove = (e: MouseEvent) => {
        mousePosRef.current.target.x = e.clientX / window.innerWidth;
        mousePosRef.current.target.y = 1 - e.clientY / window.innerHeight; // Invert Y
    };

    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', handleMouseMove);
    resize();

    let globalStartTime = Date.now();
    const render = () => {
      // Smoothly interpolate mouse position for a softer effect
      mousePosRef.current.current.x += (mousePosRef.current.target.x - mousePosRef.current.current.x) * 0.05;
      mousePosRef.current.current.y += (mousePosRef.current.target.y - mousePosRef.current.current.y) * 0.05;

      const transition = transitionRef.current;
      if (transition.startTime) {
        const elapsedTime = Date.now() - transition.startTime;
        const progress = Math.min(elapsedTime / transition.duration, 1.0);
        
        const lerp = (a: number[], b: number[], t: number): number[] => a.map((val, i) => val + (b[i] - val) * t);

        colorsRef.current = {
          color1: lerp(transition.from.color1, transition.to.color1, progress),
          color2: lerp(transition.from.color2, transition.to.color2, progress),
          color3: lerp(transition.from.color3, transition.to.color3, progress),
        };

        if (progress >= 1.0) {
          transition.startTime = null;
        }
      }

      const currentTime = (Date.now() - globalStartTime) / 1000;
      
      gl.useProgram(program);

      gl.enableVertexAttribArray(positionAttributeLocation);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.position);
      gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

      gl.uniform2f(resolutionLocation, gl.canvas.width, gl.canvas.height);
      gl.uniform1f(timeLocation, currentTime * 0.08);
      gl.uniform2f(mouseLocation, mousePosRef.current.current.x, mousePosRef.current.current.y);
      gl.uniform3fv(color1Location, colorsRef.current.color1);
      gl.uniform3fv(color2Location, colorsRef.current.color2);
      gl.uniform3fv(color3Location, colorsRef.current.color3);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      animationFrameIdRef.current = requestAnimationFrame(render);
    };
    
    render();

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameIdRef.current);
      if (glRef.current && buffersRef.current.position) {
          glRef.current.deleteBuffer(buffersRef.current.position);
          glRef.current.deleteProgram(program);
          glRef.current.deleteShader(fragmentShader);
          glRef.current.deleteShader(vertexShader);
      }
    };
  }, []);


  return <canvas ref={canvasRef} className="fixed top-0 left-0 w-full h-full -z-10" />;
};

export default WebGLBackground;