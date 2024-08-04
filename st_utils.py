import streamlit as st

class StState(object):
    def __init__(self, name: str, default) -> None:
        self.name = name
        self.default = default
        self.initialize()
    
    def initialize(self):
        if self.name not in st.session_state:
            self.reset()
    
    def reset(self):
        st.session_state[self.name] = self.default