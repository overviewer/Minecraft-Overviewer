<?
/**
 * Player Avatars for brownan's Overviewer
 *
 * @author eth0 <eth0@eth0.uk.net>
 * @author billiam
 * @version 0.3
 * @copyright Copyright (c) 2010, eth0
 *
 */

require_once('Cache/Lite.php');
require_once('classes/Char_Image.php');

define("DEBUG", false);
define("TMPDIR", '/tmp/');
define("LIFETIME", 86400);      // People don't generally change their skin more than daily

$CACHE_OPTIONS = array(
        'cacheDir' => TMPDIR,
        'lifeTime' => LIFETIME,
        'automaticSerialization' => true
);

$s = 3;
$borderWidth = 0;
$borderColor = '#fff';

if ( ! empty($_GET['player'])) {
    $player = preg_replace('/[^a-zA-Z0-9_]/', '', $_GET['player']);
    $custom_player = 'http://minecraft.net/skin/'. $player .'.png';
}

//use local default file instead of pulling from remote server.
$default_player = 'char.png'; //http://minecraft.net/img/char.png

if ( ! empty($_GET['s'])) {
    $s = (float)$_GET['s'];
    $percent = $s > 0 ? $s : 3;
}

if ( ! empty($_GET['format'])) {
    $format = $_GET['format'];
}

if ( ! empty($_GET['bw'])) {
    $borderWidth = (int)$_GET['bw'];
}

if ( ! empty($_GET['bc'])) {
    if ( preg_match('/#?[0-9a-f]{3,6}$/i', $_GET['bc'])) {
        $borderColor = (string)$_GET['bc'];
    }
}

$paramHash = md5($player . $borderWidth . $borderColor . $s . $format);

// Let's dip in to the cache and see if we have a return visitor
$Cache_Lite = new Cache_Lite($CACHE_OPTIONS);

//check for cached image with input parameters
if ( ! $generatedImage = $Cache_Lite->get($paramHash, 'avatar')) {
    //check for cached skin
    if ( ! $playerSkin = $Cache_Lite->get($player)) {

        //reduce default timeout
        $ctx = stream_context_create(array(
            'http' => array(
                'timeout' => 3,
            )
        ));

        if ( ! empty($player))
            $playerSkin = @file_get_contents($custom_player, 0, $ctx);
            
        if ( $playerSkin ) {
            $Cache_Lite->save($player, $playerSkin);

        } else {
            // Oh no custom skin? Guess we'll use the default
            $playerSkin = @file_get_contents($default_player);
            if (DEBUG) $DEBUG_TEXT = "Skin Cache Miss: " . (time() - $Cache_Lite->lastModified());
        }
    } else {
        if (DEBUG) $DEBUG_TEXT = "Skin Cache Hit: " . (time() - $Cache_Lite->lastModified());
    }

    $char = new Char_Image($playerSkin);

    $generatedImage = $char->setScale($s)
                  ->setBorder($borderColor, $borderWidth)
                  ->setFormat($format)
                  ->getImage();

    $Cache_Lite->save($generatedImage, $paramHash, 'avatar');

    if (DEBUG) $DEBUG_TEXT = "Image Cache Miss: " . (time() - $Cache_Lite->lastModified());
} else {
    if (DEBUG) $DEBUG_TEXT = "Image Cache Hit: " . (time() - $Cache_Lite->lastModified());
}
echo $DEBUG_TEXT;
Char_Image::outputImage($generatedImage);

?>
