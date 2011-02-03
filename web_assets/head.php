<?php
header('Content-type: image/png');

$player = $_GET['player'];
$usage = $_GET['usage'];

function CreateBlankPNG($w, $h)
{
    $im = imagecreatetruecolor($w, $h);
    imagesavealpha($im, true);
    $transparent = imagecolorallocatealpha($im, 0, 0, 0, 127);
    imagefill($im, 0, 0, $transparent);
    return $im;
}

function LoadPNG($imgname)
{
    /* Attempt to open */
    $im = @imagecreatefrompng($imgname);

    /* See if it failed */
    if(!$im)
    {
		$im = imagecreatefrompng('./char.png');
    }

    return $im;
}

$img = LoadPNG('http://www.minecraft.net/skin/'.$player.'.png');

if($usage == 'list')
{
	$myhead = imagecreate(15, 15);
	$color = imagecolorallocate($myhead, 250, 250, 250);
	imagefill($myhead, 0, 0, $color);

	imagecopyresized($myhead, $img, 1, 1, 8, 8, 13, 13, 8, 8);
}

if($usage == 'marker')
{
  $myhead = CreateBlankPNG(21, 27);
  $mymarker = imagecreatefrompng('./marker_empty.png');
  
  imagecopy($myhead, $mymarker, 0, 0, 0, 0, 21, 27);
  imagecopyresized($myhead, $img, 1, 1, 8, 8, 19, 20, 8, 8);
}

if($usage == 'info')
{
	$myhead = CreateBlankPNG(48, 96);
	
	imagecopyresized($myhead, $img, 12,0,8,8,24,24,8,8);
	imagecopyresized($myhead, $img, 12,24,20,20,24,26,8,12);
	imagecopyresized($myhead, $img, 12,50,0,20,12,26,4,12);
	imagecopyresized($myhead, $img, 24,50,8,20,12,26,4,12);
	imagecopyresized($myhead, $img, 2,24,44,20,10,26,4,12);
	imagecopyresized($myhead, $img, 36,24,52,20,10,26,4,12);

	imagecopyresized($myhead, $img, 6,6,32,10,6,3,2,1);
	imagecopyresized($myhead, $img, 36,6,32,10,6,3,2,1);
}

imagepng($myhead);
imagedestroy($img);
imagedestroy($myhead);
?>

