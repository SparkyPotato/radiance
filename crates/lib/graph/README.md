# radiance-graph

A fully featured, flexible, and opinionated Vulkan render graph.

The opinions enforced throughout `radiance` are:

- Bindless is used for all shader resource access.
- Images are always exclusive to a single queue family, while buffers are always shared.
