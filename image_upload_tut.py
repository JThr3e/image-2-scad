from js import document, console, Uint8Array, window, File, Blob
from pyodide.ffi import create_proxy
import asyncio
import io
import cv2
import numpy as np

from solid2 import translate, resize, polygon, scad_render
#from solid2.splines import catmull_rom_polygon, bezier_polygon
import sys

from PIL import Image, ImageFilter

async def _upload_change_and_show(e):
    #Get the first file from upload
    file_list = e.target.files
    first_item = file_list.item(0)

    #Get the data from the files arrayBuffer as an array of unsigned bytes
    array_buf = Uint8Array.new(await first_item.arrayBuffer())

    #BytesIO wants a bytes-like object, so convert to bytearray first
    bytes_list = bytearray(array_buf)
    my_bytes = io.BytesIO(bytes_list) 

    #Create PIL image from np array
    my_image = Image.open(my_bytes)

    #Log some of the image data for testing
    console.log(f"{my_image.format= } {my_image.width= } {my_image.height= }")

    # Now that we have the image loaded with PIL, we can use all the tools it makes available. 
    # "Emboss" the image, rotate 45 degrees, fill with dark green
    my_image = my_image.filter(ImageFilter.EMBOSS).resize((300,300))

    #Convert Pillow object array back into File type that createObjectURL will take
    my_stream = io.BytesIO()
    my_image.save(my_stream, format="PNG")

    #Create a JS File object with our data and the proper mime type
    image_file = File.new([Uint8Array.new(my_stream.getvalue())], "new_image_file.png", {type: "image/png"})

    #Create new tag and insert into page
    new_image = document.createElement('img')
    new_image.src = window.URL.createObjectURL(image_file)
    document.getElementById("output_upload").appendChild(new_image)

def resize_image(image, max_size):
    """
    Resize an image while keeping its aspect ratio.

    Args:
        image (numpy array): The input image.
        max_size (int): The maximum size of the resized image.

    Returns:
        numpy array: The resized image.
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if w > h:
        new_w = max_size
        new_h = int(max_size / aspect_ratio)
    else:
        new_h = max_size
        new_w = int(max_size * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

def save_file(content, filename):
    """
    Saves the given content as a file to the user's system using the File System Access API.
    """
    blob = Blob.new([content], {type: "text/plain"})

    handle = window.showSaveFilePicker({
        types: [{
            description: "Text files",
            accept: {"text/plain": [".scad"]}
        }]
    })

    writable = handle.createWritable()
    writable.write(blob)
    writable.close()

            
async def _image_to_scad(e):
    console.log("Attempted file upload: " + e.target.value)
    file_list = e.target.files
    first_item = file_list.item(0)
    #Get the data from the files arrayBuffer as an array of unsigned bytes
    array_buf = Uint8Array.new(await first_item.arrayBuffer())

    #BytesIO wants a bytes-like object, so convert to bytearray first
    bytes_list = bytearray(array_buf)
    my_bytes = io.BytesIO(bytes_list) 

    #Create PIL image from np array
    my_image = Image.open(my_bytes)

    rgb_img = np.array(my_image)
    
    rgb_img = resize_image(rgb_img, 300)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape

    white_padding = np.zeros((50, width, 3))
    white_padding[:, :] = [255, 255, 255]
    rgb_img = np.row_stack((white_padding, rgb_img))

    gray_img = 255 - gray_img
    gray_img[gray_img > 100] = 255
    gray_img[gray_img <= 100] = 0
    black_padding = np.zeros((50, width))
    gray_img = np.row_stack((black_padding, gray_img))

    gray_img_copy = np.uint8(gray_img)

    edges = cv2.Canny(gray_img_copy, 100, 200)

    result = np.zeros_like(edges)
    contours, _ = cv2.findContours(gray_img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cntr in contours:
        cv2.drawContours(result, [cntr], 0, 255, 1)

    def nearest_neighbor_path(points):
        path = [points[0]]
        remaining_points = points[1:]
        while remaining_points:
            last_point = path[-1]
            distances = [np.sqrt((p[0] - last_point[0])**2 + (p[1] - last_point[1])**2) for p in remaining_points]
            closest_index = np.argmin(distances)
            path.append(remaining_points.pop(closest_index))
        return path

    points = []
    for x,lst in enumerate(result):
        for y,elem in enumerate(lst):
            if elem == 255:
                points.append((x,y))

    path_nn = nearest_neighbor_path(points)
    
    shape = polygon(path_nn)
    resize_shape = resize([25, 0], auto=True)(shape)
    scad_txt = scad_render(resize_shape)
    save_file(scad_txt, "test.scad")
    
    pil_image = Image.fromarray(result).convert('RGB')
    #Convert Pillow object array back into File type that createObjectURL will take
    my_stream = io.BytesIO()
    pil_image.save(my_stream, format="png")

    #Create a JS File object with our data and the proper mime type
    image_file = File.new([Uint8Array.new(my_stream.getvalue())], "./test1.jpg", {type: "image/png"})
    
    new_image = document.createElement('img')
    new_image.src = window.URL.createObjectURL(image_file)
    document.getElementById("output_upload").appendChild(new_image)

# Run image processing code above whenever file is uploaded    
upload_file = create_proxy(_image_to_scad)
document.getElementById("file-upload").addEventListener("change", upload_file)
