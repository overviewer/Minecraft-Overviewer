--TEST--
Cache_Lite::Cache_Lite (automatic serialization on)
--FILE--
<?php

require_once('callcache.inc');
require_once('tmpdir.inc');
require_once('cache_lite_base.inc');

$options = array(
    'cacheDir' => tmpDir() . '/',
    'lifeTime' => 60,
    'automaticSerialization' => true
);

$Cache_Lite = new Cache_Lite($options);
multipleCallCache('array');

?>
--GET--
--POST--
--EXPECT--
==> First call (cache should be missed)
Cache Missed !
a:4:{i:0;a:2:{i:0;s:3:"foo";i:1;s:3:"bar";}i:1;i:1;i:2;s:3:"foo";i:3;s:3:"bar";}
Done !

==> Second call (cache should be hit)
Cache Hit !
a:4:{i:0;a:2:{i:0;s:3:"foo";i:1;s:3:"bar";}i:1;i:1;i:2;s:3:"foo";i:3;s:3:"bar";}
Done !

==> Third call (cache should be hit)
Cache Hit !
a:4:{i:0;a:2:{i:0;s:3:"foo";i:1;s:3:"bar";}i:1;i:1;i:2;s:3:"foo";i:3;s:3:"bar";}
Done !

==> We remove cache
Done !

==> Fourth call (cache should be missed)
Cache Missed !
a:4:{i:0;a:2:{i:0;s:3:"foo";i:1;s:3:"bar";}i:1;i:1;i:2;s:3:"foo";i:3;s:3:"bar";}
Done !

==> #5 Call with another id (cache should be missed)
Cache Missed !
a:4:{i:0;a:2:{i:0;s:3:"foo";i:1;s:3:"bar";}i:1;i:1;i:2;s:3:"foo";i:3;s:3:"bar";}
Done !

==> We remove cache
Done !
