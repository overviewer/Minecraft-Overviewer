<?
/**
 * Player Avatars for brownan's Overviewer
 * 
 * @author eth0 <eth0@eth0.uk.net>
 * @version 0.3
 * @copyright Copyright (c) 2010, eth0
 *
 */

require_once('Cache/Lite.php');

define("DEBUG", false);
define("TMPDIR", '/tmp/');
define("LIFETIME", 86400);	// People don't generally change their skin more than daily


$CACHE_OPTIONS = array(
	'cacheDir' => TMPDIR,
	'lifeTime' => LIFETIME,	
	'automaticSerialization' => true
);

$player = (string) htmlentities($_GET['player'], ENT_QUOTES, 'UTF-8');
$custom_player = 'http://minecraft.net/skin/'. $player .'.png';
$default_player = 'http://minecraft.net/img/char.png';
$s = (float) htmlentities($_GET['s'], ENT_QUOTES, 'UTF-8');
$percent = (!empty($s)) ? $s : 3;

// Let's dip in to the cache and see if we have a return visitor
$Cache_Lite = new Cache_Lite($CACHE_OPTIONS);
if ($player_skin_data = ($Cache_Lite->get($player))) {
	if (DEBUG) $DEBUG_TEXT = "Cache Hit: " . (time() - $Cache_Lite->lastModified($player));
} else {
	$player_skin_data = file_get_contents($custom_player);
	// Oh no custom skin? Guess we'll use the default
	if ( !$player_skin_data ) $player_skin_data = file_get_contents($default_player);
	if (DEBUG) $DEBUG_TEXT = "Cache Miss: " . (time() - $Cache_Lite->lastModified($player));
	$Cache_Lite->save(($player_skin_data));
}
$player_skin = imagecreatefromstring($player_skin_data);

// We get the skin dimensions and scaling factor
$width = imagesx($player_skin);
$height= imagesy($player_skin);
$new_width = $width * $percent;
$new_height = $height * $percent;

// Setup a transparent canvas to compose the head/face & helmet/face pieces
$imgPlayer = imagecreatetruecolor(8*$percent, 8*$percent);
$color = imagecolortransparent($imgPlayer, imagecolorallocatealpha($imgPlayer, 0, 0, 0, 127));
imagefill($imgPlayer, 0, 0, $color);
imagesavealpha($imgPlayer, true);
$imgPlayerHead = imagecreatefromstring($player_skin_data);
$imgPlayerFace = imagecreatefromstring($player_skin_data);
imagealphablending($imgPlayer, true);
imagealphablending($imgPlayerHead,true); 
imagealphablending($imgPlayerFace,true); 

// Copy and scale the head/face piece to canvas
imagecopyresampled($imgPlayer, $imgPlayerHead, -8*$percent, -8*$percent, 0, 0, $new_width, $new_height, $width, $height);

// Does the player have a helmet?
// We have to detect if the 'face' is entire painted the same as the background.
$rgb = imagecolorat( $imgPlayerFace, 0, 0 );
$bg_colors = imagecolorsforindex( $imgPlayerFace, $rgb );
$hasFace = false;
for ($xPix=40; $xPix <= 47; $xPix++)
{
	for ($yPix=8; $yPix <= 15; $yPix++)
	{
		$rgb = imagecolorat( $imgPlayerFace, $xPix, $yPix );
		$colors = imagecolorsforindex( $imgPlayerFace, $rgb );
		if ( count(array_diff_assoc( $colors, $bg_colors )) )
		{
			$hasFace = true;
			break 2;
		}
	}
}

// Copy and scale the helmet/face piece to canvas
if ($hasFace) imagecopyresampled($imgPlayer, $imgPlayerFace, -40*$percent, -8*$percent, 0, 0, $new_width, $new_height, $width, $height);

if (DEBUG) imagestring($imgPlayer, 2, 0, 0, $DEBUG_TEXT, imagecolorallocate($imgPlayer, 0, 0, 255));

// And finally output the image
header('Content-type: image/png');
imagepng($imgPlayer, null, 9);
imagedestroy($imgPlayer);
?>