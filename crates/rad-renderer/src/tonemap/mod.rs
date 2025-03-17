// TODO: come back to this later, with more thought towards display mapping and a uniform
// perceptual space before mapping to SDR and HDR displays and their respective gamuts/brightness
// levels.
// https://alextardif.com/Tonemapping.html

pub mod agx;
pub mod agx_hdr;
pub mod exposure;
pub mod frostbite;
pub mod null;
pub mod tony_mc_mapface;
