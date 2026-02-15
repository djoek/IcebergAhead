from model import BoundedTitanicNet
import torch
import mido
import numpy as np

import streamlit as st

st.set_page_config(layout="wide")


@st.cache_resource
def get_model():
    # 1. Create an instance of your model
    _model = BoundedTitanicNet()
    _checkpoint = torch.load('./titanic_model.pth', map_location='cpu', weights_only=False)
    _model.load_state_dict(_checkpoint['model_state_dict'])
    _model.eval()
    return _model, _checkpoint

@st.cache_resource
def get_ports(_midi_controller):
    in_port = mido.open_input(_midi_controller)
    out_port = mido.open_output(_midi_controller)
    return in_port, out_port

midi_controller = st.selectbox(label="Pick your midi controller", options=mido.get_input_names(), index=None)
if midi_controller:

    status = st.status("Getting ready!", expanded=False)
    model, checkpoint = get_model()
    status.update(label="Loaded model", state="running")

    layers =[
        model.layer1, model.layer2, model.layer3, model.layer4
    ]

    in_port, out_port = get_ports(midi_controller)

    status.update(label="Updating midi controller", state="running")
    for i, layer in enumerate(layers):
        offset = i * 16
        for j, value in enumerate((model.layer1.weight.data.flatten() * 32.0 + 64.0).int()):
            cc_message = mido.Message('control_change', channel=0, control=offset+j, value=int(value))
            out_port.send(cc_message)


    status.update(label="listening for midi", state="running")
    c1, c2, c3, c4 = st.columns(4, gap="small")

    with c1:
        layer1_df = st.empty()
    with c2:
        layer2_df = st.empty()
    with c3:
        layer3_df = st.empty()
    with c4:
        layer4_df = st.empty()

    def update_df():
        layer1_df.dataframe(layers[0].weight.data, width="content")
        layer2_df.dataframe(layers[1].weight.data, width="content")
        layer3_df.dataframe(layers[2].weight.data, width="content")
        layer4_df.dataframe(layers[3].weight.data, width="content")

    update_df()

    done = st.button("Done")

    @st.fragment(run_every=0.1)
    def listen_midi_until_done():
        if not done:
                with torch.no_grad():
                    for message in in_port.iter_pending():
                        if message.type == 'control_change' and message.channel == 0:
                            # One unravel call gives you all three indices
                            layer_idx, row, col = np.unravel_index(message.control, (4, 4, 4))
                            mapped_value = (message.value - 64) / 32
                            layers[layer_idx].weight[row, col] = mapped_value
                            update_df()

    listen_midi_until_done()

    if done:
        status.update(label="Done!", state="complete")

        in_port.close()
        out_port.close()

        from model import test_cases

        model.eval()
        scaler = checkpoint['scaler']

        with torch.no_grad():
            for case in test_cases:
                test_input = scaler.transform([case['features']])
                test_tensor = torch.FloatTensor(test_input)
                survival_prob = model(test_tensor).item()
                st.write(f"{case['desc']:40s} → {survival_prob:.2%} survival")