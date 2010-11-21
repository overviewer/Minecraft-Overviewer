--TEST--
Cache_Lite::Cache_Lite (automaticCleaning)
--FILE--
<?php

require_once('callcache.inc');
require_once('tmpdir.inc');
require_once('cache_lite_base.inc');

$options = array(
    'cacheDir' => tmpDir() . '/',
    'lifeTime' => 2,
    'automaticCleaningFactor' => 1
);

$Cache_Lite = new Cache_Lite($options);
callCache('31415926');
echo("\n");
callCache('31415926');
echo("\n");
callCache('31415926bis');
echo("\n");
callCache('31415926bis');
echo("\n");
sleep(4);
callCache('31415926'); // '31415926bis' will be cleaned
echo "\n";
$dh = opendir(tmpDir());
while ($file = readdir($dh)) {
    if (($file != '.') && ($file != '..')) {
        if (substr($file, 0, 6)=='cache_') {  
            echo "$file\n"; 
        }
    }
}

$Cache_Lite->remove('31415926');
$Cache_Lite->remove('31415926bis');

?>
--GET--
--POST--
--EXPECT--
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Cache Hit !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Cache Hit !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
cache_c21f969b5f03d33d43e04f8f136e7682_e9982ec5ca981bd365603623cf4b2277
