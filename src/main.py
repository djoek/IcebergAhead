from model import BoundedTitanicNet, test_cases

import torch
import mido
import numpy as np

import streamlit as st

st.set_page_config(layout="wide")


@st.cache_resource
def get_model(model_path='./titanic_model.pth'):
    # 1. Create an instance of your model
    _model = BoundedTitanicNet()
    _checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
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

    status = st.status("Getting ready...", expanded=False)
    model, checkpoint = get_model()
    scaler = checkpoint['scaler']

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


    status.update(label="Listening for midi", state="running")

    st.write("Adjust the weights of the layers to change their behavior and observe the results in the test cases below.")
    st.write("MIDI Channel: 1, CC Control: 0-63, Value Range: 0-127 (mapped to [-2.0, +2.0]")

    # UI elements bootstrap
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        layer1_df = st.empty()
    with c2:
        layer2_df = st.empty()
    with c3:
        layer3_df = st.empty()
    with c4:
        layer4_df = st.empty()

    results_container = st.empty()

    def show_layers():
        with layer1_df.container():
            st.subheader("Hidden Layer 1")
            st.dataframe(layers[0].weight.data, width="content", hide_index=True, )

        with layer2_df.container():
            st.subheader("Hidden Layer 2")
            st.dataframe(layers[1].weight.data, width="content", hide_index=True)

        with layer3_df.container():
            st.subheader("Hidden Layer 3")
            st.dataframe(layers[2].weight.data, width="content", hide_index=True)

        with layer4_df.container():
            st.subheader("Hidden Layer 4")
            st.dataframe(layers[3].weight.data, width="content", hide_index=True)


    @st.fragment(run_every=0.1)
    def listen_midi():
        with torch.no_grad():
            for message in in_port.iter_pending():
                if message.type == 'control_change' and message.channel == 0:
                    status.update(label=f"Received midi Ch:{message.channel} CC:{message.control}", state="running")
                    # One unravel call gives you all four layers' indices
                    layer_idx, row, col = np.unravel_index(message.control, (4, 4, 4))
                    mapped_value = (message.value - 64) / 32
                    layers[layer_idx].weight[row, col] = mapped_value
                    show_layers()

    @st.fragment(run_every=2.0)
    def show_test_cases():
        model.eval()
        with torch.no_grad():
            with results_container.container():
                st.subheader("Titanic Survival Prediction")
                for case in test_cases:
                    test_input = scaler.transform([case['features']])
                    test_tensor = torch.FloatTensor(test_input)
                    # Forward pass
                    survival_prob = model(test_tensor).item()

                    st.metric(label=f"{case['desc']:40s}", value=survival_prob, format="percent")

    listen_midi()
    show_layers()
    show_test_cases()

    done = st.button("Done")
    if done:
        status.update(label="Done!", state="complete",)

        in_port.close()
        out_port.close()

        st.stop()

