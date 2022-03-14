# SIMD Raycaster 

Voxel octree raycaster written using SIMD Intrinsics. All math is 
fixed-point. Uses AVX2, casts 8 rays simultaniously.

Raycaster code at src/game.c:343

Made this purely to practice manually vectorizing complicated semi-branchy 
algorithms, and to see just how slow this kind of raycasting is on a CPU, when 
all possible tricks are used. 

Raycasting is done all with fixed-point integers. No division is used either as
AVX2 lacks that insturction. Fuzziness and certain artifacts are due to these 
optimizations and workarounds. The algorithm appears to be severly memory 
bandwidth limited, it does not scale at all past 4 parallel rays.

Uses my Voxplat engine as a base, thus has a lot of inherited code. 
Uses OpenGL for text rendering and to present and scale the raycasted image. 
Traversal algorithm is original, 2D JS demo used to develop it at https://kosshi.net/projects/quadtree-caster/

## Images
Processor used is AMD Ryzen 1600. All cores used. 

![Bunnies](img/bunnies1.png?raw=true)
![Inside a bunny](img/bunnies2.png?raw=true)

And when things go wrong ..

![Generation bug](img/glitch1.png?raw=true)
![Distortion](img/glitch2.png?raw=true)
