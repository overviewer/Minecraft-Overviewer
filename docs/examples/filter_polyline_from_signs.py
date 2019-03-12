borders = {}

def bordersFromSigns(poi):
    """
        BORDER
         2/9
    Hanoko Village
        purple
    """
    global borders
    if poi['id'] == 'Sign' or poi['id'] == 'minecraft:sign':
        if poi['Text1'].lower() == 'border':
            # Get the order of this point and the total number of points from the 2nd line
            loc = poi['Text2']  # e.g. 2/9
            delimiter = loc.index('/')
            order = loc[:delimiter]
            size = loc[delimiter:delimiter]
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
            for border in borders:
                if border.name == name:
                    first = False
                    borders[name]['count'] += 1

            if first:
                # Create base dict for that border
                borders[name] = {
                    'color': 'red',
                    'text': name,
                    'size': -1,
                    'count': 1,
                    'polyline': {}
                }

            borders[name]['polyline'][order] = {'x': poi['x'], 'y': poi['y'], 'z': poi['z']}
            borders[name]['size'] = size

            # If color occurs once on any sign it's enough
            if color != '':
                borders[name]['color'] = color

            # If last
            if borders[name]['count'] == borders[name]['size']:
                # Convert dict to ordered list
                tuples = sorted(borders[name]['polyline'].items())
                # Make sure the signs close by adding the first sign xyz again at the end
                tuples.append(tuples[0])
                return {
                    'color': borders[name]['color'],
                    'text': borders[name]['text'],
                    'polyline': [t[1] for t in tuples]  # Only return the xyz values
                }