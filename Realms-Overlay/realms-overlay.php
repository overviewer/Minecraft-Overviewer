<?
header("Content-type: application/javascript");
$MC_PATH = "/home/eth0/minecraft";
$realm_file_path = $MC_PATH . "/plugins/Realms Files"; 
$realm_polygons = $realm_file_path . "/polygons.csv";
$realm_permissions = $realm_file_path . "/permissions.csv";
$homes = $MC_PATH . "/homes.txt";


$row = 1;
if (($handle = fopen($homes, "r")) !== FALSE) {
echo "var homeData=[\n";
    while (($data = fgetcsv($handle, 1000, ":")) !== FALSE) {
        $num = count($data);
        $row++;
        	if ($data[0] == 'everyone' && $data[1] == 'enter' && $data[3] == '0')
						$private_zones[] = $data[2];
//       for ($c=0; $c < $num; $c++) {
#					echo $data[$c] ."~";
echo '	{"player": "'. $data[0] .'", "x": '. $data[1] .', "y": '. $data[2] .', "z": '. $data[3] .', '. "},\n";
//        }
    }
echo "];\n";
    fclose($handle);
}

#echo "#####\n";



$row = 1;
$private_zones = NULL;
if (($handle = fopen($realm_permissions, "r")) !== FALSE) {
$data = fgetcsv($handle, 1000, ",");	// Thing we'll nom nom nom the header, kaithxbye
    while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) {
        $num = count($data);
        $row++;
        	if ($data[0] == 'everyone' && $data[1] == 'enter' && $data[3] == '0')
						$private_zones[] = $data[2];
    }
    fclose($handle);
}

#echo "#####\n";

$row = 1;
if (($handle = fopen($realm_polygons, "r")) !== FALSE) {
$data = fgetcsv($handle, 1000, ",");	// Thing we'll nom nom nom the header, kaithxbye
echo "var regionData=[\n\n";
    while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) {
        $num = count($data);
        $row++;
        echo "	// Zone: {$data[0]}\n";
        echo "	//   ZoneCeiling: {$data[1]}\n";
        echo "	//   ZoneFloor: {$data[2]}\n";
if (in_array($data[0], $private_zones))	{
				echo '	{"name": "'. $data[0] .'", "color": "#FF4400", "opacity": 0.5, "closed": true, "path": [' . "\n";
} else {
				echo '	{"name": "'. $data[0] .'", "color": "#AAFF00", "opacity": 0.75, "closed": true, "path": [' . "\n";
}
        for ($c=3; $c < $num; $c+=3) {
//{"x": -529, "y": 80, "z": -35},
            echo '	{"x": '. $data[$c] .', "y": '. $data[$c+1] .', "z": '. $data[$c+2] ."},\n";

        }
        echo "	]},\n\n";
    }
echo "];\n";
    fclose($handle);
}

?>
