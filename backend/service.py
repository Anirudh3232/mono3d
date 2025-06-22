// ... existing code ...
            # Post-process the mesh to reduce artifacts and improve quality
            logger.info(f"Applying Taubin smoothing to the mesh (iterations={smoothing_iterations})...")
            if smoothing_iterations > 0:
                filter_taubin(mesh, iterations=smoothing_iterations)

            # Process the mesh to fix potential issues before UV unwrapping
            logger.info("Processing mesh to fix potential issues...")
            mesh.process()

            # Use xatlas to generate UVs
            import xatlas
            import trimesh
// ... existing code ...
