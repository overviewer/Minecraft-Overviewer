<?
$custom_player = 'http://minecraft.net/skin/'. $_GET['player'] .'.png';
$default_player = 'http://eth0.uk.net/minecraft/map/char.png';
$percent = ($_GET['s']) ? $_GET['s'] : 3;

$player = getimagesize($custom_player) ? $custom_player : $default_player;

// Get new dimensions
list($width, $height) = getimagesize($player);
$new_width = $width * $percent;
$new_height = $height * $percent;

// Resample
$image_p = imagecreatetruecolor(8*$percent, 8*$percent);
//$image_p = imagecreatetruecolor(640, 320);
$image_helmet_p = imagecreatefrompng($player);
$image = imagecreatefrompng($player);

// Helmet
/*
imagealphablending($image_p, false);
$color = imagecolortransparent($image_p, imagecolorallocatealpha($image_p, 0, 0, 0, 127));
imagefill($image_p, 0, 0, $color);
imagesavealpha($image_p, true);
imagecopymerge($image_p, $image,0, 0,0, 0, 640, 320,	100); 
*/

#imagecopymerge($image_p, $image, -40*10, -8*10, 0, 0, 640, 320,	imagesx($image),imagesy($image)); 
        
        imagecopyresampled($image_p, $image, -8*$percent, -8*$percent, 0, 0, $new_width, $new_height, $width, $height);
#imagecopyresampled($image_p, $image, -40*$percent, -8*$percent, 0, 0, $new_width, $new_height, $width, $height);

// Output
header('Content-type: image/png');
imagepng($image_p, null, 1);
imagedestroy($image_p);

?>