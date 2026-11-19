---
title: "psiEngine: a Vulkan engine for looking at GPU physics"
date: 2026-09-07 09:00:00 +0300
categories: [Projects, Vulkan]
cover: psiengine-cover.png
description: "psiEngine: a C++23 Vulkan engine with GPU compute, a typed node graph and Lua scripting for visualising quantum physics computed on the GPU."
summary: Visual quantum physics compute engine. Vulkan rendering, GPU compute, a typed node graph and Lua scripting, built to draw hydrogen orbitals
---
## What it is

psiEngine is a small engine for looking at physics I compute on the GPU. Vulkan does the rendering, the GPU does the compute, and a node graph sits in the middle so I can wire a simulation up without recompiling every time I want to change a constant.

It is not a game engine and I have no plans to make it one. It exists because every time I write a kernel that produces a cloud of numbers I want to see the cloud, rotate it, poke at the parameters and watch it change, and I got tired of doing that through matplotlib and a screenshot. The first real workload is the hydrogen atom: give it (n, l, m) and it draws the orbital.

Source: [github.com/nnamu-cl/psiEngine](https://github.com/nnamu-cl/psiEngine)

## How it is put together

**Rendering is Vulkan.** SDL3 owns the window and input, volk loads the API, and shaders are written in [Slang](https://shader-slang.org/) and compiled at runtime, so a shader edit is a restart and not a rebuild. The main scene layer is GPU driven: a compute pass builds the indirect draw commands, and the vertex work for lines, circles and orbital samples happens in compute shaders before anything reaches the graphics pipeline.

**Compute runs on the GPU too.** Inside the engine that is Vulkan compute. The heavy numerical work, sampling electron densities at the bandwidth roof, is CUDA and lives in its own repo, [cuda-orbital-sampler](https://github.com/nnamu-cl/cuda-orbital-sampler), where I can profile it properly. The engine is the place those samples get drawn.

**Everything in the scene is a component.** A tiny ECS (component store, type ids) and a set of components: Transform, MeshRenderer, LineRenderer, VolumeRenderer, Atom (the quantum numbers, validated so l &lt; n and \|m\| &lt;= l) and AtomVisualizer (how those numbers become geometry). Components draw their own inspector UI.

**The node graph drives values.** A typed node system (float, int, vec2 to vec4, mat4) with value, math, vector, object and graph nodes, drawn on top of imgui\_node\_editor. Node outputs bind to component properties, so a sine node wired into a transform is an animation, and a constant node wired into an atom is a slider that changes the orbital.

**Projects save and load.** A project registry, window state and scene serialisation through glaze JSON.

**Two applications sit on top.** The psi app is the one I actually use, with a world viewport for placing and manipulating objects with gizmos and a graph editor for wiring the node graph. A second app, decisions, is a side experiment: a cashflow scenario engine scripted in Lua through sol2. It shares the window library and nothing else.

## Stack

C++23, CMake, Vulkan, SDL3, Slang, Dear ImGui, imgui\_node\_editor, ImGuizmo, im3d, ImPlot, GLM, glaze, KTX, sol2 and Lua. Developed on Linux with Wayland; a Windows path exists but has not been built in a while.

## Status

Work in progress and it moves in bursts. The rendering path, node graph, saving and the atom components work. What I am building towards is the CUDA sampler feeding the visualizer directly rather than through dumps, and a proper simulation loop on the world layer.
