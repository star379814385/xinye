from PIL import Image


class PadResize(object):
    def __init__(self,size):
        self.interpolation = Image.BILINEAR
        self.padding_v = [124, 116, 104]
        self.size = size

    def __call__(self, img):
        target_size = self.size
        padding_v = tuple(self.padding_v)
        interpolation = self.interpolation
        w, h = img.size
        if w > h:
            img = img.resize((int(target_size), int(h * target_size * 1.0 / w)), interpolation)
        else:
            img = img.resize((int(w * target_size * 1.0 / h), int(target_size)), interpolation)

        ret_img = Image.new("RGB", (target_size, target_size), padding_v)
        w, h = img.size
        st_w = int((ret_img.size[0] - w) / 2.0)
        st_h = int((ret_img.size[1] - h) / 2.0)
        ret_img.paste(img, (st_w, st_h))
        return ret_img