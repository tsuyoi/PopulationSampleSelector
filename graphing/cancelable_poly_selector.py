from matplotlib.widgets import PolygonSelector


class CancelablePolySelector(PolygonSelector):
    def __init__(self, ax, onselect, on_esc, useblit=False,
                 lineprops=None, markerprops=None, vertex_select_radius=15):
        super().__init__(ax, onselect, useblit, lineprops, markerprops, vertex_select_radius)
        self.on_esc = on_esc

    def _on_key_release(self, event):
        super()._on_key_release(event)
        if event.key == self.state_modifier_keys.get('clear'):
            self.on_esc()
