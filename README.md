# IcebergAhead

Demo script that allows you to tweak the weights of a pytorch NN using a midi controller.
So far this is only tested with a DJ TechTools MIDI Fighter Twister, but something like a FaderFox EC4 should work well

## Technical background

The demo model has four hidden layers with each four neurons, which gives you a 4x4x4 layout. 
For completeness, there are eight input neurons and one sigmoid output representing the survival chance. 
That 4x4x4 is mapped onto 64x1, which then expects you to send MIDI CC values on Channel 1, Control 0-63.
This is interpreted as bipolar, so MIDI CC value 0 is mapped to a weight of -2.0, 64 is 0.0, and 127 is just shy of +2.0

## Install

This should be straightforward, clone the repo, then cd into the dir, and 
```shell
# uv sync
```

## Train the model

```shell
# python3 ./src/model.py
```

## Run the demo

```shell
# streamlit run ./src/main.py
```

