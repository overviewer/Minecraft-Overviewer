--TEST--
Cache_Lite::Cache_Lite_Output (classical)
--FILE--
<?php

require_once('callcache.inc');
require_once('tmpdir.inc');
require_once('cache_lite_output_base.inc');

$options = array(
    'cacheDir' => tmpDir() . '/',
    'lifeTime' => 60
);

$Cache_Lite_Output = new Cache_Lite_Output($options);
multipleCallCache2();

?>
--GET--
--POST--
--EXPECT--
==> First call (cache should be missed)
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789Cache Missed !

Done !

==> Second call (cache should be hit)
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789Cache Hit !

Done !

==> Third call (cache should be hit)
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789Cache Hit !

Done !

==> We remove cache
Done !

==> Fourth call (cache should be missed)
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789Cache Missed !

Done !

==> #5 Call with another id (cache should be missed)
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789Cache Missed !

Done !

==> We remove cache
Done !
