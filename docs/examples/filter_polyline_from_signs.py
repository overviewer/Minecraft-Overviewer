polylines = {}

def lineFromSigns(poi):
    """
    Example sign:
    
    LINE
    2/8
    Train Track 2
    black
    """
    global polylines
    if poi['id'] == 'Sign' or poi['id'] == 'minecraft:sign':
        if poi['Text1'].lower() == 'line':
            # Get the order of this point and the total number of points from the 2nd line
            loc = poi['Text2']  # e.g. 2/8
            delimiter = loc.index('/')
            order = loc[:delimiter]
            size = loc[delimiter + 1:]
            try:
                order = int(order)
                size = int(size)
            except ValueError:
                return  # Skip this marker

            # Name on the 3rd line
            name = poi['Text3']
            poi['name'] = name

            # Color on the 4th line
            color = poi['Text4'].lower()

            first = True
            for key in polylines:
                if key == name:
                    first = False
                    polylines[name]['count'] += 1

            if first:
                # Create base dict for that border
                polylines[name] = {
                    'color': 'red',
                    'text': name,
                    'size': -1,
                    'count': 1,
                    'polyline': {}
                }

            polylines[name]['polyline'][order] = {'x': poi['x'], 'y': poi['y'], 'z': poi['z']}
            polylines[name]['size'] = size

            # If color occurs once on any sign it's enough
            if color != '':
                polylines[name]['color'] = color

            # If last
            if polylines[name]['count'] == polylines[name]['size']:
                # Convert dict to ordered list
                tuples = sorted(polylines[name]['polyline'].items())
                return dict(color=polylines[name]['color'], fill=False, text=polylines[name]['text'], polyline=[t[1] for t in tuples])