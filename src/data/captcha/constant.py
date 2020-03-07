FONTS = ['FONT_HERSHEY_COMPLEX',  'FONT_HERSHEY_DUPLEX',
         'FONT_HERSHEY_SIMPLEX',  'FONT_HERSHEY_TRIPLEX', 'FONT_ITALIC']

LETTER_NUM = 4

APPEARED_LETTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z'
]

CAPTCHA_TO_CATEGORY = dict(zip(APPEARED_LETTERS, range(len(APPEARED_LETTERS))))
CATEGORY_TO_CAPTCHA = dict(zip(range(len(APPEARED_LETTERS)), APPEARED_LETTERS))