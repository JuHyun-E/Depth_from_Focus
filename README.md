# Computer_Vision
## Depth from Focus
[Dataset](https://github.com/JuHyun-E/Computer_Vision/tree/master/data)
***
TASK1: Image alignment
- Feature based alignment

TASK2: Initial depth from focus measure   
>1. Blur Estimation at Edges
* an edge
   + a step function in intensity
* the blur of this edge
   + a Gaussian blurring kernel
* Multiscale edge detector
   + output: a sparse set of pixels

>2. Focus Measure
* Laplacian

>3. Cost Volume

TASK3: All-in-focus image

TASK4: Graph-cuts and weighted median filter (Depth Refinement)   
>4. Multi-label optimization using graph-cuts   
* Graph?
   + Nodes
      - usually pixels
      - sometimes samples
   + Edges
      - weights associated (W(i,j))
      - E.g. RGB value difference   
* Cut?
   + Each "cut" -> points, W(i,j)
   + Optimization problem
   + W(i,j) = |RGB(i) - RGB(j)|
   
>5. Weighed mediean filter

***
Applications: Background Stylization, Refocusing, 3D Parallax, Artistic Effect ...
