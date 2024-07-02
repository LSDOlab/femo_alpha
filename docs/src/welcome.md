# Welcome to FEMO

The **F**inite **E**lements for **M**ultidisciplinary **O**ptimization (**FEMO**) project is a general framework for incorporating partial differential equation (PDE)-based models into gradient-based optimization of multidisciplinary systems by integrating FEniCSx with the recently-developed Computational System Design Language (CSDL). We use CSDL’s abstractions to link together sub-models representing different disciplines, and use FEniCSx to compute partial derivatives of problem residuals for the PDE subsystems. CSDL can combine the derivatives of all disciplines using the chain rule and the adjoint method. The development of this framework is motivated by the problem of optimizing designs of electric vertical takeoff and landing (eVTOL) aircraft where, due to the relative novelty of this class of vehicle, there is currently a large, unexplored design space. With the generality of this framework, FEMO can also be used to facilitate research on a wide-range of PDE-constrained MDO problems beyond eVTOL applications.

![pde_mdo_problem](/src/images/femo_opt_new.png "A general PDE-constrained MDO problem")

# Cite us
```none

@misc{xiang2024,
    author = "Xiang, Ru 
            and van Schie, Sebastiaan P.C.
            and Scotzniovsky, Luca 
            and Yan, Jiayao
            and Kamensky, David 
            and Hwang, John T.",
    title  = "Automating adjoint sensitivity analysis for multidisciplinary models involving partial differential equations",
    howpublished = {Jul 2024 (under review)}
}

@misc{scotzniovsky2024,
    author = "Scotzniovsky, Luca 
            and Xiang, Ru 
            and Cheng, Zeyu 
            and Rodriguez, Gabriel 
            and Kamensky, David 
            and Mi, Chris 
            and Hwang, John T.",
    title  = "Geometric Design of Electric Motors Using Adjoint-based Shape Optimization",
    howpublished = {Feb 2024, Preprint available at \url{https://doi.org/10.21203/rs.3.rs-3941981/v1}}
}
```

<!-- Remove/add custom pages from/to toc as per your package's requirement -->

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/examples
src/api
```
