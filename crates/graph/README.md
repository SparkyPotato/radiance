# radiance-graph

A flexible and opinionated Vulkan render graph and initialization toolkit.

The opinions enforced throughout `radiance` are:
- BDA is used for all buffer access.
- Bindless is used for all image and sampler access.
- All resources are always shared across all queue families.
- Only GPUs with three queue families (graphics, compute, transfer) are supported.
