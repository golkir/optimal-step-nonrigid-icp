## Optimal Step Non-Rigid ICP for Surface Registration

### Introduction


In this repo, we implement optimal step non-rigid ICP registration algorithm based on [Amberg et al](https://www.researchgate.net/publication/200172513_Optimal_Step_Nonrigid_ICP_Algorithms_for_Surface_Registration).

Inspired by the [pytorch-nicp] https://github.com/wuhaozhe/pytorch-nicp repository, we add support for
full pose scan registration using Mediapipe pose landmarks. In addition, we use `open3d` mesh library instead of `pytorch3d` which makes it easier to run code on CPU. 

The codebase is a work in progress and the following features are to be added in the future releases:
- sampling of mesh vertices for faster performance, experimenting with different sampling methods
- efficient face registration 

------

### Quick Start
#### Install
We use python3.11. The code is tested on macOS Monterey 12.7.5.  

To use the code:

1. download [Mediapipe pose landmarker model](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task) and put it into `models` folder.

2. Put template and target meshes you want to use in the `data` folder and specify paths to them in `optimal_step_nicp/demo.py`
3. To install the package, run

```
pip install .
```
#### Demo

- For demo, run 
```
optimal-step-nicp
```
or 
```
python optimal_step_nicp/demo.py
```

### Configuration

- **inner_iter**: Number of inner loop iterations. Affects non-rigid smoothing.
- **outer_iter**: Number of outer loop iterations.
- **log_iter**: Frequency of logging loss.
- **milestones**: Milestones for learning rate adjustment. Default: `[50, 80, 100, 110, 120, 130, 140]`.
- **stiffness_weights**: Controls the stiffness of transformation. Determines how far the close points can deform. Default: `[50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]`.
- **landmark_weights**: Controls the significance of the landmarks term. Default: `[5, 2, 0.5, 0, 0, 0, 0, 0]`.
- **laplacian_weight**: Laplacian smoothing weight. Default: `250`.


### Algorithms steps

1. Collect pose landmarks for template and target meshes
2. Calculate the rigid transformation from the template to the target landmarks using SVD-based alignment.
3. **In Outer Loop**:
   1. **Local Affine Transformation + Stiffness Term**: Use a neural network with a learnable linear layer to learn the transformation and calculate stiffness, ensuring that close points have similar transformations.
   2. Deform the template based on the transformation.
   3. Find the closest points between the deformed template and the target.
4. **Run Inner Loop**:
   1. Calculate the vertex distance loss, ensuring that only vertices with a distance less than `threshold` are accounted for.
   2. Calculate the landmark loss, which is the distance between target landmarks and transformed template landmarks, weighted by landmark weights.
   3. Calculate the stiffness loss.
   4. Calculate the Laplacian loss, which measures how the vertex differs from its neighbors.
   5. Sum up all the losses.
   6. Perform backpropagation.
   7. Recalculate the local affine transformation in the inner loop.
5. **Calculate Final Distance**: In the outer loop, calculate the final distance loss and apply transformed vertices to the template mesh. 

The registration is done! 😊🎉


## Credits

We thank to the authors of the [pytorch-nicp](https://github.com/wuhaozhe/pytorch-nicp) who developed efficient pytorch3d-based implementation and demonstrated it on the BFM model for face registration. 

This project uses a template mesh provided by [Meshcapade](https://meshcapade.com/). We thank them for their valuable resources. [License](https://app.box.com/s/mdx2m368j9m0jgkkjnf67l6blrwrt20f/file/822907587054 )
