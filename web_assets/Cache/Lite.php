<?php

/**
* Fast, light and safe Cache Class
*
* Cache_Lite is a fast, light and safe cache system. It's optimized
* for file containers. It is fast and safe (because it uses file
* locking and/or anti-corruption tests).
*
* There are some examples in the 'docs/examples' file
* Technical choices are described in the 'docs/technical' file
*
* Memory Caching is from an original idea of
* Mike BENOIT <ipso@snappymail.ca>
*
* Nota : A chinese documentation (thanks to RainX <china_1982@163.com>) is
* available at :
* http://rainx.phpmore.com/manual/cache_lite.html
*
* @package Cache_Lite
* @category Caching
* @version $Id: Lite.php,v 1.54 2009/07/07 05:34:37 tacker Exp $
* @author Fabien MARTY <fab@php.net>
*/

define('CACHE_LITE_ERROR_RETURN', 1);
define('CACHE_LITE_ERROR_DIE', 8);

class Cache_Lite
{

    // --- Private properties ---

    /**
    * Directory where to put the cache files
    * (make sure to add a trailing slash)
    *
    * @var string $_cacheDir
    */
    var $_cacheDir = '/tmp/';

    /**
    * Enable / disable caching
    *
    * (can be very usefull for the debug of cached scripts)
    *
    * @var boolean $_caching
    */
    var $_caching = true;

    /**
    * Cache lifetime (in seconds)
    *
    * If null, the cache is valid forever.
    *
    * @var int $_lifeTime
    */
    var $_lifeTime = 3600;

    /**
    * Enable / disable fileLocking
    *
    * (can avoid cache corruption under bad circumstances)
    *
    * @var boolean $_fileLocking
    */
    var $_fileLocking = true;

    /**
    * Timestamp of the last valid cache
    *
    * @var int $_refreshTime
    */
    var $_refreshTime;

    /**
    * File name (with path)
    *
    * @var string $_file
    */
    var $_file;
    
    /**
    * File name (without path)
    *
    * @var string $_fileName
    */
    var $_fileName;

    /**
    * Enable / disable write control (the cache is read just after writing to detect corrupt entries)
    *
    * Enable write control will lightly slow the cache writing but not the cache reading
    * Write control can detect some corrupt cache files but maybe it's not a perfect control
    *
    * @var boolean $_writeControl
    */
    var $_writeControl = true;

    /**
    * Enable / disable read control
    *
    * If enabled, a control key is embeded in cache file and this key is compared with the one
    * calculated after the reading.
    *
    * @var boolean $_writeControl
    */
    var $_readControl = true;

    /**
    * Type of read control (only if read control is enabled)
    *
    * Available values are :
    * 'md5' for a md5 hash control (best but slowest)
    * 'crc32' for a crc32 hash control (lightly less safe but faster, better choice)
    * 'strlen' for a length only test (fastest)
    *
    * @var boolean $_readControlType
    */
    var $_readControlType = 'crc32';

    /**
    * Pear error mode (when raiseError is called)
    *
    * (see PEAR doc)
    *
    * @see setToDebug()
    * @var int $_pearErrorMode
    */
    var $_pearErrorMode = CACHE_LITE_ERROR_RETURN;
    
    /**
    * Current cache id
    *
    * @var string $_id
    */
    var $_id;

    /**
    * Current cache group
    *
    * @var string $_group
    */
    var $_group;

    /**
    * Enable / Disable "Memory Caching"
    *
    * NB : There is no lifetime for memory caching ! 
    *
    * @var boolean $_memoryCaching
    */
    var $_memoryCaching = false;

    /**
    * Enable / Disable "Only Memory Caching"
    * (be carefull, memory caching is "beta quality")
    *
    * @var boolean $_onlyMemoryCaching
    */
    var $_onlyMemoryCaching = false;

    /**
    * Memory caching array
    *
    * @var array $_memoryCachingArray
    */
    var $_memoryCachingArray = array();

    /**
    * Memory caching counter
    *
    * @var int $memoryCachingCounter
    */
    var $_memoryCachingCounter = 0;

    /**
    * Memory caching limit
    *
    * @var int $memoryCachingLimit
    */
    var $_memoryCachingLimit = 1000;
    
    /**
    * File Name protection
    *
    * if set to true, you can use any cache id or group name
    * if set to false, it can be faster but cache ids and group names
    * will be used directly in cache file names so be carefull with
    * special characters...
    *
    * @var boolean $fileNameProtection
    */
    var $_fileNameProtection = true;
    
    /**
    * Enable / disable automatic serialization
    *
    * it can be used to save directly datas which aren't strings
    * (but it's slower)    
    *
    * @var boolean $_serialize
    */
    var $_automaticSerialization = false;
    
    /**
    * Disable / Tune the automatic cleaning process
    *
    * The automatic cleaning process destroy too old (for the given life time)
    * cache files when a new cache file is written.
    * 0               => no automatic cache cleaning
    * 1               => systematic cache cleaning
    * x (integer) > 1 => automatic cleaning randomly 1 times on x cache write
    *
    * @var int $_automaticCleaning
    */
    var $_automaticCleaningFactor = 0;
    
    /**
    * Nested directory level
    *
    * Set the hashed directory structure level. 0 means "no hashed directory 
    * structure", 1 means "one level of directory", 2 means "two levels"... 
    * This option can speed up Cache_Lite only when you have many thousands of 
    * cache file. Only specific benchs can help you to choose the perfect value 
    * for you. Maybe, 1 or 2 is a good start.
    *
    * @var int $_hashedDirectoryLevel
    */
    var $_hashedDirectoryLevel = 0;
    
    /**
    * Umask for hashed directory structure
    *
    * @var int $_hashedDirectoryUmask
    */
    var $_hashedDirectoryUmask = 0700;
    
    /**
     * API break for error handling in CACHE_LITE_ERROR_RETURN mode
     * 
     * In CACHE_LITE_ERROR_RETURN mode, error handling was not good because
     * for example save() method always returned a boolean (a PEAR_Error object
     * would be better in CACHE_LITE_ERROR_RETURN mode). To correct this without
     * breaking the API, this option (false by default) can change this handling.
     * 
     * @var boolean
     */
    var $_errorHandlingAPIBreak = false;
    
    // --- Public methods ---

    /**
    * Constructor
    *
    * $options is an assoc. Available options are :
    * $options = array(
    *     'cacheDir' => directory where to put the cache files (string),
    *     'caching' => enable / disable caching (boolean),
    *     'lifeTime' => cache lifetime in seconds (int),
    *     'fileLocking' => enable / disable fileLocking (boolean),
    *     'writeControl' => enable / disable write control (boolean),
    *     'readControl' => enable / disable read control (boolean),
    *     'readControlType' => type of read control 'crc32', 'md5', 'strlen' (string),
    *     'pearErrorMode' => pear error mode (when raiseError is called) (cf PEAR doc) (int),
    *     'memoryCaching' => enable / disable memory caching (boolean),
    *     'onlyMemoryCaching' => enable / disable only memory caching (boolean),
    *     'memoryCachingLimit' => max nbr of records to store into memory caching (int),
    *     'fileNameProtection' => enable / disable automatic file name protection (boolean),
    *     'automaticSerialization' => enable / disable automatic serialization (boolean),
    *     'automaticCleaningFactor' => distable / tune automatic cleaning process (int),
    *     'hashedDirectoryLevel' => level of the hashed directory system (int),
    *     'hashedDirectoryUmask' => umask for hashed directory structure (int),
    *     'errorHandlingAPIBreak' => API break for better error handling ? (boolean)
    * );
    *
    * @param array $options options
    * @access public
    */
    function Cache_Lite($options = array(NULL))
    {
        foreach($options as $key => $value) {
            $this->setOption($key, $value);
        }
    }
    
    /**
    * Generic way to set a Cache_Lite option
    *
    * see Cache_Lite constructor for available options
    *
    * @var string $name name of the option
    * @var mixed $value value of the option
    * @access public
    */
    function setOption($name, $value) 
    {
        $availableOptions = array('errorHandlingAPIBreak', 'hashedDirectoryUmask', 'hashedDirectoryLevel', 'automaticCleaningFactor', 'automaticSerialization', 'fileNameProtection', 'memoryCaching', 'onlyMemoryCaching', 'memoryCachingLimit', 'cacheDir', 'caching', 'lifeTime', 'fileLocking', 'writeControl', 'readControl', 'readControlType', 'pearErrorMode');
        if (in_array($name, $availableOptions)) {
            $property = '_'.$name;
            $this->$property = $value;
        }
    }
    
    /**
    * Test if a cache is available and (if yes) return it
    *
    * @param string $id cache id
    * @param string $group name of the cache group
    * @param boolean $doNotTestCacheValidity if set to true, the cache validity won't be tested
    * @return string data of the cache (else : false)
    * @access public
    */
    function get($id, $group = 'default', $doNotTestCacheValidity = false)
    {
        $this->_id = $id;
        $this->_group = $group;
        $data = false;
        if ($this->_caching) {
            $this->_setRefreshTime();
            $this->_setFileName($id, $group);
            clearstatcache();
            if ($this->_memoryCaching) {
                if (isset($this->_memoryCachingArray[$this->_file])) {
                    if ($this->_automaticSerialization) {
                        return unserialize($this->_memoryCachingArray[$this->_file]);
                    }
                    return $this->_memoryCachingArray[$this->_file];
                }
                if ($this->_onlyMemoryCaching) {
                    return false;
                }                
            }
            if (($doNotTestCacheValidity) || (is_null($this->_refreshTime))) {
                if (file_exists($this->_file)) {
                    $data = $this->_read();
                }
            } else {
                if ((file_exists($this->_file)) && (@filemtime($this->_file) > $this->_refreshTime)) {
                    $data = $this->_read();
                }
            }
            if (($data) and ($this->_memoryCaching)) {
                $this->_memoryCacheAdd($data);
            }
            if (($this->_automaticSerialization) and (is_string($data))) {
                $data = unserialize($data);
            }
            return $data;
        }
        return false;
    }
    
    /**
    * Save some data in a cache file
    *
    * @param string $data data to put in cache (can be another type than strings if automaticSerialization is on)
    * @param string $id cache id
    * @param string $group name of the cache group
    * @return boolean true if no problem (else : false or a PEAR_Error object)
    * @access public
    */
    function save($data, $id = NULL, $group = 'default')
    {
        if ($this->_caching) {
            if ($this->_automaticSerialization) {
                $data = serialize($data);
            }
            if (isset($id)) {
                $this->_setFileName($id, $group);
            }
            if ($this->_memoryCaching) {
                $this->_memoryCacheAdd($data);
                if ($this->_onlyMemoryCaching) {
                    return true;
                }
            }
            if ($this->_automaticCleaningFactor>0 && ($this->_automaticCleaningFactor==1 || mt_rand(1, $this->_automaticCleaningFactor)==1)) {
				$this->clean(false, 'old');			
			}
            if ($this->_writeControl) {
                $res = $this->_writeAndControl($data);
                if (is_bool($res)) {
                    if ($res) {
                        return true;  
                    }
                    // if $res if false, we need to invalidate the cache
                    @touch($this->_file, time() - 2*abs($this->_lifeTime));
                    return false;
                }            
            } else {
                $res = $this->_write($data);
            }
            if (is_object($res)) {
                // $res is a PEAR_Error object 
                if (!($this->_errorHandlingAPIBreak)) {   
                    return false; // we return false (old API)
                }
            }
            return $res;
        }
        return false;
    }

    /**
    * Remove a cache file
    *
    * @param string $id cache id
    * @param string $group name of the cache group
    * @param boolean $checkbeforeunlink check if file exists before removing it
    * @return boolean true if no problem
    * @access public
    */
    function remove($id, $group = 'default', $checkbeforeunlink = false)
    {
        $this->_setFileName($id, $group);
        if ($this->_memoryCaching) {
            if (isset($this->_memoryCachingArray[$this->_file])) {
                unset($this->_memoryCachingArray[$this->_file]);
                $this->_memoryCachingCounter = $this->_memoryCachingCounter - 1;
            }
            if ($this->_onlyMemoryCaching) {
                return true;
            }
        }
        if ( $checkbeforeunlink ) {
            if (!file_exists($this->_file)) return true;
        }
        return $this->_unlink($this->_file);
    }

    /**
    * Clean the cache
    *
    * if no group is specified all cache files will be destroyed
    * else only cache files of the specified group will be destroyed
    *
    * @param string $group name of the cache group
    * @param string $mode flush cache mode : 'old', 'ingroup', 'notingroup', 
    *                                        'callback_myFunction'
    * @return boolean true if no problem
    * @access public
    */
    function clean($group = false, $mode = 'ingroup')
    {
        return $this->_cleanDir($this->_cacheDir, $group, $mode);
    }
       
    /**
    * Set to debug mode
    *
    * When an error is found, the script will stop and the message will be displayed
    * (in debug mode only). 
    *
    * @access public
    */
    function setToDebug()
    {
        $this->setOption('pearErrorMode', CACHE_LITE_ERROR_DIE);
    }

    /**
    * Set a new life time
    *
    * @param int $newLifeTime new life time (in seconds)
    * @access public
    */
    function setLifeTime($newLifeTime)
    {
        $this->_lifeTime = $newLifeTime;
        $this->_setRefreshTime();
    }

    /**
    * Save the state of the caching memory array into a cache file cache
    *
    * @param string $id cache id
    * @param string $group name of the cache group
    * @access public
    */
    function saveMemoryCachingState($id, $group = 'default')
    {
        if ($this->_caching) {
            $array = array(
                'counter' => $this->_memoryCachingCounter,
                'array' => $this->_memoryCachingArray
            );
            $data = serialize($array);
            $this->save($data, $id, $group);
        }
    }

    /**
    * Load the state of the caching memory array from a given cache file cache
    *
    * @param string $id cache id
    * @param string $group name of the cache group
    * @param boolean $doNotTestCacheValidity if set to true, the cache validity won't be tested
    * @access public
    */
    function getMemoryCachingState($id, $group = 'default', $doNotTestCacheValidity = false)
    {
        if ($this->_caching) {
            if ($data = $this->get($id, $group, $doNotTestCacheValidity)) {
                $array = unserialize($data);
                $this->_memoryCachingCounter = $array['counter'];
                $this->_memoryCachingArray = $array['array'];
            }
        }
    }
    
    /**
    * Return the cache last modification time
    *
    * BE CAREFUL : THIS METHOD IS FOR HACKING ONLY !
    *
    * @return int last modification time
    */
    function lastModified() 
    {
        return @filemtime($this->_file);
    }
    
    /**
    * Trigger a PEAR error
    *
    * To improve performances, the PEAR.php file is included dynamically.
    * The file is so included only when an error is triggered. So, in most
    * cases, the file isn't included and perfs are much better.
    *
    * @param string $msg error message
    * @param int $code error code
    * @access public
    */
    function raiseError($msg, $code)
    {
        include_once('PEAR.php');
        return PEAR::raiseError($msg, $code, $this->_pearErrorMode);
    }
    
    /**
     * Extend the life of a valid cache file
     * 
     * see http://pear.php.net/bugs/bug.php?id=6681
     * 
     * @access public
     */
    function extendLife()
    {
        @touch($this->_file);
    }
    
    // --- Private methods ---
    
    /**
    * Compute & set the refresh time
    *
    * @access private
    */
    function _setRefreshTime() 
    {
        if (is_null($this->_lifeTime)) {
            $this->_refreshTime = null;
        } else {
            $this->_refreshTime = time() - $this->_lifeTime;
        }
    }
    
    /**
    * Remove a file
    * 
    * @param string $file complete file path and name
    * @return boolean true if no problem
    * @access private
    */
    function _unlink($file)
    {
        if (!@unlink($file)) {
            return $this->raiseError('Cache_Lite : Unable to remove cache !', -3);
        }
        return true;        
    }

    /**
    * Recursive function for cleaning cache file in the given directory
    *
    * @param string $dir directory complete path (with a trailing slash)
    * @param string $group name of the cache group
    * @param string $mode flush cache mode : 'old', 'ingroup', 'notingroup',
                                             'callback_myFunction'
    * @return boolean true if no problem
    * @access private
    */
    function _cleanDir($dir, $group = false, $mode = 'ingroup')     
    {
        if ($this->_fileNameProtection) {
            $motif = ($group) ? 'cache_'.md5($group).'_' : 'cache_';
        } else {
            $motif = ($group) ? 'cache_'.$group.'_' : 'cache_';
        }
        if ($this->_memoryCaching) {
	    foreach($this->_memoryCachingArray as $key => $v) {
                if (strpos($key, $motif) !== false) {
                    unset($this->_memoryCachingArray[$key]);
                    $this->_memoryCachingCounter = $this->_memoryCachingCounter - 1;
                }
            }
            if ($this->_onlyMemoryCaching) {
                return true;
            }
        }
        if (!($dh = opendir($dir))) {
            return $this->raiseError('Cache_Lite : Unable to open cache directory !', -4);
        }
        $result = true;
        while ($file = readdir($dh)) {
            if (($file != '.') && ($file != '..')) {
                if (substr($file, 0, 6)=='cache_') {
                    $file2 = $dir . $file;
                    if (is_file($file2)) {
                        switch (substr($mode, 0, 9)) {
                            case 'old':
                                // files older than lifeTime get deleted from cache
                                if (!is_null($this->_lifeTime)) {
                                    if ((time() - @filemtime($file2)) > $this->_lifeTime) {
                                        $result = ($result and ($this->_unlink($file2)));
                                    }
                                }
                                break;
                            case 'notingrou':
                                if (strpos($file2, $motif) === false) {
                                    $result = ($result and ($this->_unlink($file2)));
                                }
                                break;
                            case 'callback_':
                                $func = substr($mode, 9, strlen($mode) - 9);
                                if ($func($file2, $group)) {
                                    $result = ($result and ($this->_unlink($file2)));
                                }
                                break;
                            case 'ingroup':
                            default:
                                if (strpos($file2, $motif) !== false) {
                                    $result = ($result and ($this->_unlink($file2)));
                                }
                                break;
                        }
                    }
                    if ((is_dir($file2)) and ($this->_hashedDirectoryLevel>0)) {
                        $result = ($result and ($this->_cleanDir($file2 . '/', $group, $mode)));
                    }
                }
            }
        }
        return $result;
    }
      
    /**
    * Add some date in the memory caching array
    *
    * @param string $data data to cache
    * @access private
    */
    function _memoryCacheAdd($data)
    {
        $this->_memoryCachingArray[$this->_file] = $data;
        if ($this->_memoryCachingCounter >= $this->_memoryCachingLimit) {
            list($key, ) = each($this->_memoryCachingArray);
            unset($this->_memoryCachingArray[$key]);
        } else {
            $this->_memoryCachingCounter = $this->_memoryCachingCounter + 1;
        }
    }

    /**
    * Make a file name (with path)
    *
    * @param string $id cache id
    * @param string $group name of the group
    * @access private
    */
    function _setFileName($id, $group)
    {
        
        if ($this->_fileNameProtection) {
            $suffix = 'cache_'.md5($group).'_'.md5($id);
        } else {
            $suffix = 'cache_'.$group.'_'.$id;
        }
        $root = $this->_cacheDir;
        if ($this->_hashedDirectoryLevel>0) {
            $hash = md5($suffix);
            for ($i=0 ; $i<$this->_hashedDirectoryLevel ; $i++) {
                $root = $root . 'cache_' . substr($hash, 0, $i + 1) . '/';
            }   
        }
        $this->_fileName = $suffix;
        $this->_file = $root.$suffix;
    }
    
    /**
    * Read the cache file and return the content
    *
    * @return string content of the cache file (else : false or a PEAR_Error object)
    * @access private
    */
    function _read()
    {
        $fp = @fopen($this->_file, "rb");
        if ($this->_fileLocking) @flock($fp, LOCK_SH);
        if ($fp) {
            clearstatcache();
            $length = @filesize($this->_file);
            $mqr = get_magic_quotes_runtime();
            if ($mqr) {
                set_magic_quotes_runtime(0);
            }
            if ($this->_readControl) {
                $hashControl = @fread($fp, 32);
                $length = $length - 32;
            } 
            if ($length) {
                $data = @fread($fp, $length);
            } else {
                $data = '';
            }
            if ($mqr) {
                set_magic_quotes_runtime($mqr);
            }
            if ($this->_fileLocking) @flock($fp, LOCK_UN);
            @fclose($fp);
            if ($this->_readControl) {
                $hashData = $this->_hash($data, $this->_readControlType);
                if ($hashData != $hashControl) {
                    if (!(is_null($this->_lifeTime))) {
                        @touch($this->_file, time() - 2*abs($this->_lifeTime)); 
                    } else {
                        @unlink($this->_file);
                    }
                    return false;
                }
            }
            return $data;
        }
        return $this->raiseError('Cache_Lite : Unable to read cache !', -2); 
    }
    
    /**
    * Write the given data in the cache file
    *
    * @param string $data data to put in cache
    * @return boolean true if ok (a PEAR_Error object else)
    * @access private
    */
    function _write($data)
    {
        if ($this->_hashedDirectoryLevel > 0) {
            $hash = md5($this->_fileName);
            $root = $this->_cacheDir;
            for ($i=0 ; $i<$this->_hashedDirectoryLevel ; $i++) {
                $root = $root . 'cache_' . substr($hash, 0, $i + 1) . '/';
                if (!(@is_dir($root))) {
                    @mkdir($root, $this->_hashedDirectoryUmask);
                }
            }
        }
        $fp = @fopen($this->_file, "wb");
        if ($fp) {
            if ($this->_fileLocking) @flock($fp, LOCK_EX);
            if ($this->_readControl) {
                @fwrite($fp, $this->_hash($data, $this->_readControlType), 32);
            }
            $mqr = get_magic_quotes_runtime();
            if ($mqr) {
                set_magic_quotes_runtime(0);
            }
            @fwrite($fp, $data);
            if ($mqr) {
                set_magic_quotes_runtime($mqr);
            }
            if ($this->_fileLocking) @flock($fp, LOCK_UN);
            @fclose($fp);
            return true;
        }      
        return $this->raiseError('Cache_Lite : Unable to write cache file : '.$this->_file, -1);
    }
       
    /**
    * Write the given data in the cache file and control it just after to avoir corrupted cache entries
    *
    * @param string $data data to put in cache
    * @return boolean true if the test is ok (else : false or a PEAR_Error object)
    * @access private
    */
    function _writeAndControl($data)
    {
        $result = $this->_write($data);
        if (is_object($result)) {
            return $result; # We return the PEAR_Error object
        }
        $dataRead = $this->_read();
        if (is_object($dataRead)) {
            return $dataRead; # We return the PEAR_Error object
        }
        if ((is_bool($dataRead)) && (!$dataRead)) {
            return false; 
        }
        return ($dataRead==$data);
    }
    
    /**
    * Make a control key with the string containing datas
    *
    * @param string $data data
    * @param string $controlType type of control 'md5', 'crc32' or 'strlen'
    * @return string control key
    * @access private
    */
    function _hash($data, $controlType)
    {
        switch ($controlType) {
        case 'md5':
            return md5($data);
        case 'crc32':
            return sprintf('% 32d', crc32($data));
        case 'strlen':
            return sprintf('% 32d', strlen($data));
        default:
            return $this->raiseError('Unknown controlType ! (available values are only \'md5\', \'crc32\', \'strlen\')', -5);
        }
    }
    
} 

?>
