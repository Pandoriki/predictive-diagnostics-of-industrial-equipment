import React, { useRef, useEffect, useCallback } from 'react';
import { Theme } from '../types';

interface WebGLCardBackgroundProps {
  theme: Theme;
  parentRef: React.RefObject<HTMLElement>;
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
  uniform vec2 u_mouse;
  uniform vec3 u_bgColor;
  uniform vec3 u_borderColor;

  // 2D Random
  float random (vec2 st) {
      return fract(sin(dot(st.xy,
                          vec2(12.9898,78.233)))*
          43758.5453123);
  }

  // 2D Noise based on Morgan McGuire @morgan3d
  float noise (in vec2 st) {
      vec2 i = floor(st);
      vec2 f = fract(st);

      float a = random(i);
      float b = random(i + vec2(1.0, 0.0));
      float c = random(i + vec2(0.0, 1.0));
      float d = random(i + vec2(1.0, 1.0));

      // Smoothstep interpolation
      vec2 u = f*f*(3.0-2.0*f);
      return mix(a, b, u.x) +
              (c - a)* u.y * (1.0 - u.x) +
              (d - b) * u.x * u.y;
  }

  void main() {
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    
    // Create a parallax offset based on mouse position (from -0.5 to 0.5)
    vec2 parallax_offset = (u_mouse - 0.5) * 0.1;
    vec2 distorted_st = st - parallax_offset;

    // Create a distortion pattern using noise.
    float distortion = noise(distorted_st * 6.0) * 0.05;

    // Apply the distortion to the coordinates for the main texture.
    float pattern = noise((distorted_st + distortion) * 3.0);
    
    // Base color with a subtle pattern.
    vec3 color = u_bgColor + (pattern - 0.5) * 0.1;
    
    // Create a soft border.
    vec2 border_uv = st * 2.0 - 1.0;
    float border_dist = 1.0 - max(abs(border_uv.x), abs(border_uv.y));
    float border = smoothstep(0.0, 0.1, border_dist);
    
    color = mix(u_borderColor, color, border);
    
    float alpha = 0.4 + (1.0 - border) * 0.1;

    gl_FragColor = vec4(color, alpha);
  }
`;

const WebGLCardBackground: React.FC<WebGLCardBackgroundProps> = ({ theme, parentRef }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const animationFrameIdRef = useRef<number>(0);
  const isHoveringRef = useRef(false);
  
  const mousePosRef = useRef({ current: { x: 0.5, y: 0.5 }, target: { x: 0.5, y: 0.5 } });

  const uniformsRef = useRef<{ 
    resolution?: WebGLUniformLocation | null;
    mouse?: WebGLUniformLocation | null;
    bgColor?: WebGLUniformLocation | null;
    borderColor?: WebGLUniformLocation | null;
    positionAttribute?: number;
  }>({});

  const colorsRef = useRef({
    bgColor: [0.08, 0.1, 0.15],
    borderColor: [0.1, 0.3, 0.4],
  });

  const themeColors = {
    [Theme.Dark]: {
      bgColor: [0.08, 0.1, 0.15],
      borderColor: [0.1, 0.3, 0.4],
    },
    [Theme.Light]: {
      bgColor: [0.9, 0.92, 0.98],
      borderColor: [0.7, 0.8, 0.95],
    }
  };

  const render = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const uniforms = uniformsRef.current;
    if (!gl || !program || uniforms.positionAttribute === undefined) return;
    
    gl.useProgram(program);

    gl.enableVertexAttribArray(uniforms.positionAttribute);
    gl.vertexAttribPointer(uniforms.positionAttribute, 2, gl.FLOAT, false, 0, 0);

    gl.uniform2f(uniforms.resolution, gl.canvas.width, gl.canvas.height);
    gl.uniform2f(uniforms.mouse, mousePosRef.current.current.x, mousePosRef.current.current.y);
    gl.uniform3fv(uniforms.bgColor, colorsRef.current.bgColor);
    gl.uniform3fv(uniforms.borderColor, colorsRef.current.borderColor);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const parent = parentRef.current;
    if (!parent) return;

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
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    programRef.current = program;

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Shader program link error: ' + gl.getProgramInfoLog(program));
      return;
    }
    
    gl.useProgram(program);
    uniformsRef.current = {
      positionAttribute: gl.getAttribLocation(program, "a_position"),
      resolution: gl.getUniformLocation(program, "u_resolution"),
      mouse: gl.getUniformLocation(program, "u_mouse"),
      bgColor: gl.getUniformLocation(program, "u_bgColor"),
      borderColor: gl.getUniformLocation(program, "u_borderColor"),
    };

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    
    const renderLoop = () => {
        // Interpolate mouse position for smooth movement
        mousePosRef.current.current.x += (mousePosRef.current.target.x - mousePosRef.current.current.x) * 0.1;
        mousePosRef.current.current.y += (mousePosRef.current.target.y - mousePosRef.current.current.y) * 0.1;
        
        render();

        // Stop the loop if not hovering and the mouse position is close to the target (center)
        const dx = Math.abs(mousePosRef.current.current.x - mousePosRef.current.target.x);
        const dy = Math.abs(mousePosRef.current.current.y - mousePosRef.current.target.y);

        if (!isHoveringRef.current && dx < 0.001 && dy < 0.001) {
            return; // Stop animation
        }
        
        animationFrameIdRef.current = requestAnimationFrame(renderLoop);
    };

    const handleMouseEnter = () => {
        isHoveringRef.current = true;
        cancelAnimationFrame(animationFrameIdRef.current); // Ensure no duplicate loops
        animationFrameIdRef.current = requestAnimationFrame(renderLoop);
    };

    const handleMouseLeave = () => {
        isHoveringRef.current = false;
        mousePosRef.current.target = { x: 0.5, y: 0.5 };
    };

    const handleMouseMove = (e: MouseEvent) => {
        const rect = parent.getBoundingClientRect();
        mousePosRef.current.target.x = (e.clientX - rect.left) / rect.width;
        mousePosRef.current.target.y = 1.0 - (e.clientY - rect.top) / rect.height; // Invert Y
    };

    const resizeObserver = new ResizeObserver(entries => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        render();
      }
    });

    parent.addEventListener('mouseenter', handleMouseEnter);
    parent.addEventListener('mouseleave', handleMouseLeave);
    parent.addEventListener('mousemove', handleMouseMove);
    resizeObserver.observe(parent);
    
    // Initial render
    render();

    return () => {
      parent.removeEventListener('mouseenter', handleMouseEnter);
      parent.removeEventListener('mouseleave', handleMouseLeave);
      parent.removeEventListener('mousemove', handleMouseMove);
      resizeObserver.unobserve(parent);
      cancelAnimationFrame(animationFrameIdRef.current);
      if (gl) {
          gl.deleteBuffer(positionBuffer);
          gl.deleteProgram(program);
          gl.deleteShader(fragmentShader);
          gl.deleteShader(vertexShader);
      }
    };
  }, [render, parentRef]);
  
  useEffect(() => {
    // This effect handles theme changes. It simply updates the target colors.
    // The actual animation happens in the main render loop if it's active,
    // or in a temporary animation loop if it's not.
    
    const fromColors = colorsRef.current;
    const toColors = themeColors[theme];
    const duration = 500;
    let startTime: number;

    const lerp = (a: number[], b: number[], t: number) => a.map((val, i) => val + (b[i] - val) * t);

    const animateTheme = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const elapsedTime = timestamp - startTime;
      const progress = Math.min(elapsedTime / duration, 1.0);

      colorsRef.current = {
        bgColor: lerp(fromColors.bgColor, toColors.bgColor, progress),
        borderColor: lerp(fromColors.borderColor, toColors.borderColor, progress),
      };
      
      // If a mouse-driven render loop isn't active, we render manually here.
      if (!isHoveringRef.current) {
          render();
      }

      if (progress < 1.0) {
        requestAnimationFrame(animateTheme);
      }
    };

    requestAnimationFrame(animateTheme);
    
  }, [theme, render]);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full rounded-3xl -z-10" />;
};

export default WebGLCardBackground;