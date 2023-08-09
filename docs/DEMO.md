### Running the Demo
We present Demo code both to generate a single image and to generate a scene

#### Installation
Instructions [here](https://github.com/crockwell/pixelsynth/blob/master/docs/INSTALL.md#requirements-used-by-paper-code). We recommend using the environment versions used by paper code for demo examples.

#### Pretrained Model 
RealEstate pretrained model is available for download [here](https://fouheylab.eecs.umich.edu/~cnris/pixelsynth/modelcheckpoints/realestate/pixelsynth.pth). We recommend this model unless working with scanned images similar to Matterport. Model trained on Matterport is available [here](https://fouheylab.eecs.umich.edu/~cnris/pixelsynth/modelcheckpoints/mp3d/pixelsynth.pth).
Place these models in new directories modelcheckpoints/realestate and modelcheckpoints/mp3d.

#### Synthesize an Image
- sh scripts/demo_image.sh

Important Arguments: 
- old_model: model path to load. Recommend realestate model unless working with scanned images similar to Matterport.
- demo_img_name: input image name, in demo folder
- result_folder: location to save outputs
- temperature: use .5 if using 1 sample or on Matterport, .7 for 50 samples if on RealEstate10K.
- num_samples: 1 is fastest, 50 has best results
- direction: of rotation, e.g. "U" for up or "DR" for down and to the right.
- rotation: in radians; we suggest <= .6 for horizontal and <= .3 for vertical rotation

#### Synthesize a Scene
- sh scripts/demo_scene.sh

Additional Arguments:
- directions: select direction(s) in which to synthesize scene
- num_split: select number of total images to render in a given direction. 
E.g. 3 means 2 intermediate images will be synthesized, equally spaced.

#### Example Argument

Check the scripts directory
