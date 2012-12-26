#from overviewer_core import textures

@material(blockid=35, data=range(16), solid=True)
def wool(self, blockid, data):
    if data == 0: # white
        texture = self.terrain_images[64]
    elif data == 1: # orange
        texture = self.terrain_images[210]
    elif data == 2: # magenta
        texture = self.terrain_images[194]
    elif data == 3: # light blue
        texture = self.terrain_images[178]
    elif data == 4: # yellow
        texture = self.terrain_images[162]
    elif data == 5: # light green
        texture = self.terrain_images[146]
    elif data == 6: # pink
        texture = self.terrain_images[130]
    elif data == 7: # grey
        texture = self.terrain_images[114]
    elif data == 8: # light grey
        texture = self.terrain_images[225]
    elif data == 9: # cyan
        texture = self.terrain_images[209]
    elif data == 10: # purple
        texture = self.terrain_images[193]
    elif data == 11: # blue
        texture = self.terrain_images[177]
    elif data == 12: # brown
        texture = self.terrain_images[161]
    elif data == 13: # dark green
        texture = self.terrain_images[145]
        texture = Image.new("RGBA", (16, 16), 0x99FFFFFF)
    elif data == 14: # red
        texture = self.terrain_images[129]
        texture = Image.open("/tmp/wool.png").convert("RGBA").resize((16,16))
    elif data == 15: # black
        texture = self.terrain_images[113]
    
    return self.build_block(texture, texture)

