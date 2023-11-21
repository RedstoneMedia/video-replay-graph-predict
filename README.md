# Video Highlight Finder
This project uses a simple ML approach to predict video attention over time.

## Process
Numerous videos were collected, then split into sequences of 128x128 RGB images. \
These images (and also the normalized video duration) are then fed into the model.
The model is trained to predict YouTube's "most replayed graph" data based on these inputs. \
This is an openly accessible metric that produces acceptable results, without manual labeling.