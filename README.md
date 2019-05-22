# Real-time ray-tracing with Vulkan/CUDA interop.

* Real-time ray-tracing (RTRT) renderer
* Diffuse, reflection, refraction and combined rendering modes
* Renders spheres or triangle soup
* Normal/bump mapping
* University assignment for final undergraduate project
* This program is provided only as a demonstration of my software development experience

## Setup

Requirements:

| Tested GPUs | CUDA architecture | Software | Solution File to use |
| --- | --- | --- | --- |
| Nvidia geforce 940M | compute_50,sm_50 | Vulkan 1.0, CUDA 10.0, Visual Studio 2017 | simpleVulkan_vs2017.sln |
| Nvidia geforce GTX 960 | compute_52,sm_52 | Vulkan 1.1, CUDA 10.1, Visual Studio 2017 | simpleVulkan_vs2015.sln | 

* Ensure VS2017, CUDA toolkit and Vulkan are installed (Guidance [on Nvidia CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) )
* Download the project
* Unzip and open either "simpleVulkan_vs2017.sln" or "simpleVulkan_vs2015.sln"
* To build and run, choose "Release" (or "debug") and select "Local Windows Debugger"

## Instructions To Use

| Control | Key(s) |
| --- | --- |
| Change rendering mode | TAB |
| Move camera | WASD + QZ |
| Move point light | EGRF + XC |

* OBJ and image textures are easily changeable
* Settings can be changed with compile-time directives (requires rebuild, more detailed information to follow)

## Demonstrates

* Real-time ray-traced rendering
* Static BVH for spatial partitioning
* CUDA 10.0 GPGPU programming
* Vulkan/CUDA interop.
* Normal/bump mapping
* Motion blur (via temporally distributed ray-tracing)

## Screenshots

![rtrt screenshot 1](https://raw.githubusercontent.com/iamhaker23/portfolio/master/rtrt/1.PNG "Refraction render mode")
![rtrt screenshot 3](https://raw.githubusercontent.com/iamhaker23/portfolio/master/rtrt/3.PNG "Reflection render mode")
![rtrt screenshot 2](https://raw.githubusercontent.com/iamhaker23/portfolio/master/rtrt/2.PNG "Diffuse render mode")
![rtrt screenshot 4](https://raw.githubusercontent.com/iamhaker23/portfolio/master/rtrt/0.jpg "Diffuse render mode")
![rtrt screenshot 4](https://raw.githubusercontent.com/iamhaker23/portfolio/master/rtrt/4.PNG "Combined reflection/refraction (CRR) render mode")

## Credits

* **Hakeem Bux** - *Developed for final undergraduate project* - [iamhaker23](https://github.com/iamhaker23)

## Acknowledgments

* Based on some initial classes provided by UEA CMP faculty

My thanks to:

* Dr. Stephen Laycock for supervising
* Nvidia for their publicly available Vulkan/CUDA interop. "sinewave" example
* Tomas Akenine-Moller for public-domain triangle-box intersect code (used in BVH generation)
* Amy Williams, Robert Cook, Moller and Trumbore, Lengyel, Snell, Blinn and Fresnel for the mathematical "tools" which form the basis of this program.
* Scratchapixel, Realtimerendering and vulkantutorial for their invaluable learning resources.
* Cornell University for the Cornell Box 3D model, Stanford University for the Stanford Bunny 3D model and Morgan McGuire for hosting such resources online.

## Issues

* Issues and improvements are discussed in my technical report (available here soon).
