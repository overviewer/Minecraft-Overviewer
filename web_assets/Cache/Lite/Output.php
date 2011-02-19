<?php

/**
* This class extends Cache_Lite and uses output buffering to get the data to cache.
*
* There are some examples in the 'docs/examples' file
* Technical choices are described in the 'docs/technical' file
*
* @package Cache_Lite
* @version $Id: Output.php,v 1.4 2006/01/29 00:22:07 fab Exp $
* @author Fabien MARTY <fab@php.net>
*/

require_once('Cache/Lite.php');

class Cache_Lite_Output extends Cache_Lite
{

    // --- Public methods ---

    /**
    * Constructor
    *
    * $options is an assoc. To have a look at availables options,
    * see the constructor of the Cache_Lite class in 'Cache_Lite.php'
    *
    * @param array $options options
    * @access public
    */
    function Cache_Lite_Output($options)
    {
        $this->Cache_Lite($options);
    }

    /**
    * Start the cache
    *
    * @param string $id cache id
    * @param string $group name of the cache group
    * @param boolean $doNotTestCacheValidity if set to true, the cache validity won't be tested
    * @return boolean true if the cache is hit (false else)
    * @access public
    */
    function start($id, $group = 'default', $doNotTestCacheValidity = false)
    {
        $data = $this->get($id, $group, $doNotTestCacheValidity);
        if ($data !== false) {
            echo($data);
            return true;
        }
        ob_start();
        ob_implicit_flush(false);
        return false;
    }

    /**
    * Stop the cache
    *
    * @access public
    */
    function end()
    {
        $data = ob_get_contents();
        ob_end_clean();
        $this->save($data, $this->_id, $this->_group);
        echo($data);
    }

}


?>
