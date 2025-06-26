import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

interface ModelPreviewProps {
  objData: string;
}

export default function ModelPreview({ objData }: ModelPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !objData) return;

    const currentContainer = containerRef.current;
    
    // Initialize scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    // Initialize camera
    const camera = new THREE.PerspectiveCamera(
      50,
      currentContainer.clientWidth / currentContainer.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 2, 5);

    // Initialize renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(currentContainer.clientWidth, currentContainer.clientHeight);
    currentContainer.appendChild(renderer.domElement);

    // Add lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);
    
    // Add grid helper for better orientation
    const gridHelper = new THREE.GridHelper(10, 10);
    scene.add(gridHelper);

    // Add controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    // Load OBJ
    const loader = new OBJLoader();
    const object = loader.parse(objData);
    
    // Center and scale the model
    const box = new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    object.position.sub(center);
    
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 3 / maxDim; // Use a slightly larger scale
    object.scale.multiplyScalar(scale);

    scene.add(object);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      const width = currentContainer.clientWidth;
      const height = currentContainer.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (currentContainer) {
        currentContainer.removeChild(renderer.domElement);
      }
      renderer.dispose();
      // Dispose of materials and geometries to free up GPU memory
      scene.traverse(child => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach(material => material.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
    };
  }, [objData]);

  return (
    <div 
      ref={containerRef} 
      className="w-full h-full rounded-lg overflow-hidden"
    />
  );
}
