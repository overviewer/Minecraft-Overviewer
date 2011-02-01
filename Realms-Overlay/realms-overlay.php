<?
/**
 * Realms Overlay for brownan's Overviewer
 * 
 * @author eth0 <eth0@eth0.uk.net>
 * @version 0.1
 * @copyright Copyright (c) 2010, eth0
 */

$MC_PATH = "/home/eth0/minecraft";

$REALM_FILE_PATH = $MC_PATH . "/plugins/Realms Files"; 
$REALM_POLYGONS = $REALM_FILE_PATH . "/polygons.csv";
$REALM_PERMISSIONS = $REALM_FILE_PATH . "/permissions.csv";

$ZONE['color'] = "#AAFF00";
$ZONE['opacity'] = "0.75";

$PRIVATE_ZONE['color'] = "#FF4400";
$PRIVATE_ZONE['opacity'] = "0.5";

header("Content-type: application/javascript");

$row = 1;
$private_zones = NULL;
if (($handle = fopen($REALM_PERMISSIONS, "r")) !== FALSE) {
	$data = fgetcsv($handle, 1000, ",");	// Thing we'll nom nom nom the header, kaithxbye
	while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) {
		$num = count($data);
		$row++;
		if ($data[0] == 'everyone' && $data[1] == 'enter' && $data[3] == '0')
			$private_zones[] = $data[2];
	}
	fclose($handle);
}

$row = 1;
if (($handle = fopen($REALM_POLYGONS, "r")) !== FALSE) {
	$data = fgetcsv($handle, 1000, ",");	// Thing we'll nom nom nom the header, kaithxbye
	echo "var regionData=[\n\n";
    while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) {
			$num = count($data);
			$row++;
			echo "	// Zone: {$data[0]}\n";
			echo "	//   ZoneCeiling: {$data[1]}\n";
			echo "	//   ZoneFloor: {$data[2]}\n";
			if (in_array($data[0], $private_zones))	{
				echo '	{"name": "'. $data[0] .'", "color": "'. $PRIVATE_ZONE['color'] .'", "opacity": '. $PRIVATE_ZONE['opacity'] .', "closed": true, "path": [' . "\n";
			} else {
				echo '	{"name": "'. $data[0] .'", "color": "'. $ZONE['color'] .'", "opacity": '. $ZONE['opacity'] .', "closed": true, "path": [' . "\n";
			}
			for ($c=3; $c < $num; $c+=3) {
				echo '	{"x": '. $data[$c] .', "y": '. $data[$c+1] .', "z": '. $data[$c+2] ."},\n";
			}
		echo "	]},\n\n";
    }
	echo "];\n";
	fclose($handle);
}

?>
