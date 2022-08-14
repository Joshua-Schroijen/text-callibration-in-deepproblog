import io
import PIL
from PIL import Image

def figure_to_PIL(figure):
    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format='png')
    return Image.open(image_buffer)

def concat_figs_vertically(figures, filename):
    imgs = [figure_to_PIL(figure) for figure in figures]
    breakpoint()
    widths, heights = zip(*(i.size for i in imgs))
    width_of_new_image = min(widths)
    height_of_new_image = sum(heights)
    new_im = Image.new('RGB', (width_of_new_image, height_of_new_image))
    new_pos = 0
    for im in imgs:
        new_im.paste(im, (0, new_pos))
        new_pos += im.size[1]
    new_im.save(filename)
    return new_im
