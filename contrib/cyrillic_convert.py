#!/usr/bin/env python3

"""Convert gibberish back into Cyrillic"""

import fileinput
import argparse
import sys

gibberish_to_cyrillic = {
    'À': 'А',
    'Á': 'Б',
    'Â': 'В',
    'Ã': 'Г',
    'Ä': 'Д',
    'Å': 'Е',
    'Æ': 'Ж',
    'Ç': 'З',
    'È': 'И',
    'É': 'Й',
    'Ê': 'К',
    'Ë': 'Л',
    'Ì': 'М',
    'Í': 'Н',
    'Î': 'О',
    'Ï': 'П',
    'Ð': 'Р',
    'Ñ': 'С',
    'Ò': 'Т',
    'Ó': 'У',
    'Ô': 'Ф',
    'Õ': 'Х',
    'Ö': 'Ц',
    '×': 'Ч',
    'Ø': 'Ш',
    'Ù': 'Щ',
    'Ú': 'Ъ',
    'Û': 'Ы',
    'Ü': 'Ь',
    'Ý': 'Э',
    'Þ': 'Ю',
    'ß': 'Я',
    'à': 'а',
    'á': 'б',
    'â': 'в',
    'ã': 'г',
    'ä': 'д',
    'å': 'е',
    'æ': 'ж',
    'ç': 'з',
    'è': 'и',
    'é': 'й',
    'ê': 'к',
    'ë': 'л',
    'ì': 'м',
    'í': 'н',
    'î': 'о',
    'ï': 'п',
    'ð': 'р',
    'ñ': 'с',
    'ò': 'т',
    'ó': 'у',
    'ô': 'ф',
    'õ': 'х',
    'ö': 'ц',
    '÷': 'ч',
    'ø': 'ш',
    'ù': 'щ',
    'ú': 'ъ',
    'û': 'ы',
    'ü': 'ь',
    'ý': 'э',
    'þ': 'ю',
    'ÿ': 'я'
}
trans_table = {ord(k): v for k, v in gibberish_to_cyrillic.items()}


def convert(s):
    return s.translate(trans_table)


if __name__ == '__main__':
    description = """
    If you have signs that should be Cyrillic, but are instead gibberish,
    this script will convert it back to proper Cyrillic.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file', metavar='markers.js')

    args = parser.parse_args()
    convert(args.markers_file)
    for line in fileinput.input(files=markers_file, inplace=1):
        print(convert(s), end='')
