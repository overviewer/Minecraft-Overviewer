<?php
/**
 * Character image generator
 *
 * @author billiam
 * @version 0.1
 */
class Char_Image {
    const LOC_BACK = 'back';
    const LOC_FRONT = 'front';
    const LOC_OUTSIDE = 'outside';
    const LOC_LEFT = 'left';
    const LOC_RIGHT = 'right';
    const LOC_TOP = 'top';
    const LOC_BOTTOM = 'bottom';

    const FORMAT_FACE = 'face';
    const FORMAT_FLAT = 'flat';

    public static $formats = array(
        self::FORMAT_FACE,
        self::FORMAT_FLAT,
    );

    /**
     * Coordinates of body parts in character PNG
     * (incomplete)
     *
     * @var array
     * Format:
     * <body part> => array (
     *   <location> => array(x, y, width, height)
     * )
     */
    public static $coords = array(
        'head' => array(
            self::LOC_RIGHT => array(0,8,8,8),
            self::LOC_FRONT => array(8,8,8,8),
            self::LOC_LEFT  => array(16, 8,8,8),
            self::LOC_BACK  => array(24,8,8,8),
        ),
        'body' => array(
            self::LOC_FRONT => array(20,20,8,12),
        ),
        'arms' => array(
            self::LOC_FRONT => array(44,20,4,12),
        ),
        'legs' => array(
            self::LOC_FRONT => array(4,20,4,12),
        ),
        'helmet' => array(
            self::LOC_FRONT =>  array(40,8,8,8),
        )
    );
    /**
     * @var resource
     */
    protected $_skin;

    /**
     * Current image scale
     * @var int
     */
    protected $_scale = 1;

    /**
     * Border color
     *
     * Formats:
     * #aaffcc, afc
     * @var string
     */
    protected $_borderColor;

    /**
     * Border width
     * 
     * @var int
     */
    protected $_borderWidth = 0;

    /**
     * @var array
     */
    protected $_parts = array();

    /**
     * @var string
     */
    protected $_format = self::FORMAT_FACE;

    /**
     * Create image from input file
     *
     * @param string|resource $skin
     */
    public function __construct($skin)
    {
        if(is_string($skin)) {
            $this->_skin = imagecreatefromstring($skin);
        } elseif(is_resource($skin)) {
            $this->_skin = $skin;
        }
    }

    /**
     * Destroy skin image on destruct
     */
    public function __destruct()
    {
        imagedestroy($this->_skin);
    }

    
    /**
     * Add a border around output player image
     *
     * @param string $color
     * @param int $width
     * @return Char_Image
     */
    public function setBorder($color, $width)
    {
        $color = self::htmlToRgb($color);

        if ($color && is_numeric($width) && $width > 0 && $width < 100) {
            $width = (int)$width;
            $this->_borderWidth = $width;
            $this->_borderColor = $color;
        }
        return $this;
    }

    /**
     * Set format for displayed player avatars
     *
     * @see self::$formats
     * @param string $format
     * @return Char_Image
     */
    public function setFormat($format)
    {
        if (in_array($format, self::$formats)) {
            $this->_format = $format;
        }
        return $this;
    }

    /**
     * Set the image scale
     *
     * @param int $scale
     *
     * @return Char_Image
     */
    public function setScale($scale)
    {
        $this->_scale = (int)$scale;

        return $this;
    }

    /**
     * Get the image scale
     * @return int
     */
    public function getScale()
    {
        return $this->_scale;
    }

    /**
     * Does player skin include a helmet?
     *
     * @return bool
     */
    public function hasHelmet()
    {
        //check for transparent pixels in region outside hat area
       $index = imagecolorat($this->_skin, 25, 0);
       $alpha = ($index & 0x7F000000) >> 24;
       return $alpha == 127;
    }


    /**
     * Display player as flat, front view
     *
     * @param bool $fetch If true, image output will be returned as a string
     * @return void|string
     */
    public function _showFlat($fetch = false)
    {
        $face = self::$coords['head'][self::LOC_FRONT];
        $arms = self::$coords['arms'][self::LOC_FRONT];
        $body = self::$coords['body'][self::LOC_FRONT];
        $legs = self::$coords['legs'][self::LOC_FRONT];
        $helmet = self::$coords['helmet'][self::LOC_FRONT];

        $srcWidth = $arms[2] * 2 + $body[2];
        $srcHeight = $legs[3] +  $body[3] + $face[3] + 1;

        $image = $this->_createImage($srcWidth, $srcHeight);

        //Place and scale image/body parts
        if ( $this->_borderWidth) {
            $color = imagecolorallocate($image, $this->_borderColor[0], $this->_borderColor[1], $this->_borderColor[2]);
            //border behind trunk
            imagefilledrectangle($image, $arms[2] * $this->_scale, $this->_scale, ($arms[2] + $body[2]) * $this->_scale + $this->_borderWidth * 2 - 1, ($face[3] + $body[3] + $legs[3] + 1) * $this->_scale + $this->_borderWidth * 2 - 1, $color);
            imagefilledrectangle($image, 0, ($face[3] + 1) * $this->_scale, ($arms[2] * 2 + $body[2]) * $this->_scale + $this->_borderWidth * 2, (1 + $face[3] + $arms[3]) * $this->_scale + $this->_borderWidth * 2 - 1, $color);
        }

        $this->_copyTo($image, $this->_skin, $face, $arms[2], 1);
        $this->_copyTo($image, $this->_skin, $arms, 0, $face[3] + 1);
        $this->_copyTo($image, $this->_skin, $arms, $arms[2] + $body[2], $face[3] + 1, true);
        $this->_copyTo($image, $this->_skin, $body, $arms[2], $face[3] + 1);
        $this->_copyTo($image, $this->_skin, $legs, $arms[2], $face[3] + $body[3] + 1);
        $this->_copyTo($image, $this->_skin, $legs, $arms[2] + $legs[2], $face[3] + $body[3] + 1, true);

        if ( $this->hasHelmet()) {
            $this->_copyTo($image, $this->_skin, $helmet, $arms[2] - 1, 0, false, 10/8);
        }

        return self::getPngString($image);
    }

    /**
     * Display player face only
     *
     * @return void
     */
    protected function _showFace()
    {
        $face = self::$coords['head'][self::LOC_FRONT];
        $helmet = self::$coords['helmet'][self::LOC_FRONT];

        $image = $this->_createImage($face[2] + 2, $face[3] + 2);

        //Place and scale image/body parts
        if ( $this->_borderWidth) {
            $color = imagecolorallocate($image, $this->_borderColor[0], $this->_borderColor[1], $this->_borderColor[2]);
            imagefilledrectangle($image, $this->_scale, $this->_scale, ($face[3] + 1) * $this->_scale + $this->_borderWidth * 2 - 1, ($face[3] + 1) * $this->_scale + $this->_borderWidth * 2 - 1, $color);
        }

        $this->_copyTo($image, $this->_skin, $face, 1, 1);

        if ( $this->hasHelmet()) {
            $this->_copyTo($image, $this->_skin, $helmet, 0, 0, false, 10/8);
        }
        
        return self::getPngString($image);
    }

    /**
     * Copy portion of skin image to destination output image
     * 
     * @param resource $dest
     * @param resource $src
     * @param array $coords
     * @param int $destX
     * @param int $destY
     * @param bool $flip Flip image horizontally
     * @param int $partScale Body part-specific scale (ex: helmets)
     * @return void
     */
    protected function _copyTo($dest, $src, $coords, $destX, $destY, $flip=false, $partScale = 1)
    {
        $flipMult = 1;
        $widthOffset = 0;
        if ($flip) {
            $flipMult = -1;
            $widthOffset = $coords[2] -1;
        }

        imagecopyresampled(
            $dest,  //destination
            $src,   //source
            $destX * $this->_scale + $this->_borderWidth,  //destination x
            $destY * $this->_scale + $this->_borderWidth,  //destination y
            $coords[0] + $widthOffset, //source x
            $coords[1],                //source y
            $coords[2] * $this->_scale * $partScale, //destination width
            $coords[3] * $this->_scale * $partScale, //destination height
            $coords[2] * $flipMult,//source width
            $coords[3]             //source height
        );
    }

    /**
     * Get image string for an image resource
     * 
     * @static
     * @param resource $image
     * @return void
     */
    public static function getPngString($image)
    {
        ob_start();
        imagepng($image, null, 9);
        $returnImage = ob_get_contents();
        ob_end_clean();
        
        return $returnImage;

    }

    /**
     * Output image to client
     * 
     * @static
     * @param string|resource $image
     * @return void
     */
    public static function outputImage($image)
    {
        header('Content-type: image/png');
        if (is_resource($image)) {
           $image = self::getPngString($image);
        }
        
        header('Content-type: image/png');
        echo $image;
    }


    /**
     * Generate a player image resource
     * 
     * @param int $width
     * @param int $height
     * @return resource
     */
    protected function _createImage($width, $height)
    {
        $image = imagecreatetruecolor($width * $this->_scale + $this->_borderWidth * 2, $height * $this->_scale + $this->_borderWidth * 2);
        imagesavealpha($image, true);
        $transparent = imagecolorallocatealpha($image, 0, 0, 0, 127);
        imagefill($image, 0, 0, $transparent);
        imagealphablending($image,true);
        return $image;
    }

    /**
     * @param bool $return
     * @return string|void
     */
    public function getImage()
    {
        if ( $this->_format == self::FORMAT_FACE) {
            $image = $this->_showFace();
        } else {
            $image = $this->_showFlat();
        }
        return $image;
    }


    /*public function showIsometric()
    {

    }*/

    /**
     * Convert html code to rgb array
     * 
     * @static
     * @param  $color
     * @return array|bool
     */
    public static function htmlToRgb($color)
    {
        if ($color[0] == '#')
            $color = substr($color, 1);

        if (strlen($color) == 6)
            list($r, $g, $b) = array($color[0].$color[1],
                                     $color[2].$color[3],
                                     $color[4].$color[5]);
        elseif (strlen($color) == 3)
            list($r, $g, $b) = array($color[0].$color[0], $color[1].$color[1], $color[2].$color[2]);
        else
            return false;

        $r = hexdec($r); $g = hexdec($g); $b = hexdec($b);

        return array($r, $g, $b);
    }
}
