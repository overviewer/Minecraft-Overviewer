<?php

// Bench script of Cache_Lite_Output
// $Id: bench2.php,v 1.4 2002/09/28 18:05:29 fab Exp $

require_once('Cache/Lite/Output.php');

$options = array(
    'caching' => true,
    'cacheDir' => '/tmp/',
    'lifeTime' => 10
);

$cache = new Cache_Lite_Output($options);

if (!($cache->start('123'))) {
    // Cache missed...
    for($i=0;$i<1000;$i++) { // Making of the page...
        echo('0123456789');
    }
    $cache->end();
}

?>