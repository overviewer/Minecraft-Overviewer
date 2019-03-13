polygons = {}

def areaFromSigns(poi):
    """
    Example sign:
    
    AREA
    4
    Harbour
    yellow
    """
    global polygons
    if poi['id'] == 'Sign' or poi['id'] == 'minecraft:sign':
        if poi['Text1'].lower() == 'area':
            # Get the total number of points from the 2nd line (needed only once)
            try:
                size = int(poi['Text2'])
            except ValueError:
                pass

            # Name on the 3rd line (needs to be on every sign)
            name = poi['Text3']
            poi['name'] = name

            # Color on the 4th line (needed only once)
            color = poi['Text4'].lower()

            first = True
            for key in polygons:
                if key == name:
                    first = False
                    polygons[name]['count'] += 1

            if first:
                # Create base dict for that border
                polygons[name] = {
                    'color': 'red',
                    'text': name,
                    'size': -1,
                    'count': 1,
                    'polygon': []
                }

            polygons[name]['polygon'].append({'x': poi['x'], 'y': poi['y'], 'z': poi['z']})
            polygons[name]['size'] = size

            # If color occurs once on any sign it's enough
            if color != '':
                polygons[name]['color'] = color

            # If last
            if polygons[name]['count'] == polygons[name]['size']:
                return dict(color=polygons[name]['color'], fill=True, text=polygons[name]['text'], polygon=polygons[name]['polygon'])