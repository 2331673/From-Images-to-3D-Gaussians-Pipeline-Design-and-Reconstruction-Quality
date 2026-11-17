# Third party sources and data

The repository vendors the following upstream source snapshots. Their original notices and license files are retained inside each directory; vendored code is not claimed as an original implementation.

| Component | Pinned commit | Upstream license |
|---|---|---|
| [gsplat v1.5.3](https://github.com/nerfstudio-project/gsplat) | `937e29912570c372bed6747a5c9bf85fed877bae` | Apache-2.0 |
| [legacy pycolmap](https://github.com/rmbrualla/pycolmap) | `cc7ea4b7301720ac29287dbe450952511b32125e` | MIT |
| [nerfview](https://github.com/nerfstudio-project/nerfview) | `4538024fe0d15fd1a0e4d760f3695fc44ca72787` | MIT, declared in upstream pyproject.toml |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) | `328dc9836f513d00c4b5bc38fe30478b4435cbb5` | MIT |
| [GLM](https://github.com/g-truc/glm) | `33b4a621a697a305bc3a7610d290677b96beb181` | See vendored `copying.txt` |

Source archive checksums and exact upstream URLs are in `provenance/*_source.json`. The portable scripts adjust project paths; the upstream training implementation is unchanged.

The 10 objects come from the group's filtered Objaverse data. `manifests/objects.json` records asset UIDs and the source-manifest hash. This course repository includes only the small input renderings, cameras and derived results needed for these experiments; source meshes are not bundled. Upstream code licenses do not relicense the source assets or their renders.
