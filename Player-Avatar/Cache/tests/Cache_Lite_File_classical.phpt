--TEST--
Cache_Lite::Cache_Lite_File (classical)
--FILE--
<?php

require_once('callcache.inc');
require_once('tmpdir.inc');
require_once('cache_lite_file_base.inc');

$master = tmpDir() . '/' . 'foobar.masterfile';
$options = array(
    'cacheDir' => tmpDir() . '/',
    'lifeTime' => 60,
    'masterFile' => $master
);

$f = fopen($master, 'w');
fwrite($f, 'foobar');
fclose($f);
sleep(1);

$Cache_Lite = new Cache_Lite_File($options);
multipleCallCache('string');
multipleCallCache3_1('string');

echo "==> We touch masterFile\n";
touch($master);
sleep(1);
clearstatcache();
echo "\nDone !\n\n";
$Cache_Lite = new Cache_Lite_File($options);
sleep(1);
multipleCallCache3_2('string');

?>
--GET--
--POST--
--EXPECT--
==> First call (cache should be missed)
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Done !

==> Second call (cache should be hit)
Cache Hit !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Done !

==> Third call (cache should be hit)
Cache Hit !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Done !

==> We remove cache
Done !

==> Fourth call (cache should be missed)
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Done !

==> #5 Call with another id (cache should be missed)
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Done !

==> We remove cache
Done !
==> #6 call (cache should be missed)
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Done !

==> #7 call (cache should be hit)
Cache Hit !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Done !

==> We touch masterFile

Done !

==> #8 call (cache should be missed)
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Done !

==> We remove cache
Done !
