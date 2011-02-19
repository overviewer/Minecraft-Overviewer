--TEST--
Cache_Lite::Cache_Lite (error)
--FILE--
<?php

require_once('callcache.inc');
require_once('tmpdir.inc');
require_once('cache_lite_base.inc');

$options = array(
    'cacheDir' => tmpDir() . '31451992gjhgjh'. '/', # I hope there will be no directory with that silly name
    'lifeTime' => 60
);

$Cache_Lite = new Cache_Lite($options);
multipleCallCache();

?>
--GET--
--POST--
--EXPECT--
==> First call (cache should be missed)
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Error when saving cache !

Done !

==> Second call (cache should be hit)
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Error when saving cache !

Done !

==> Third call (cache should be hit)
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Error when saving cache !

Done !

==> We remove cache
Done !

==> Fourth call (cache should be missed)
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Error when saving cache !

Done !

==> #5 Call with another id (cache should be missed)
Cache Missed !
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Error when saving cache !

Done !

==> We remove cache
Done !
