$SOURCE_DIR = "1_memory_blocks"
echo "Start parsing files"

$dirs = gci $SOURCE_DIR -Directory
foreach($d in $dirs)
{
    $files = gci $d.FullName
    $count = $files.count
    $files = $files | resolve-path
    # split training and testing data to 5:1
    $test  = $files | get-random -count ([int]($count/5))
    $train = $files | ?{$_ -notin $test}     

    $path_test  = "2_image\Test\$($d.Name)"
    $path_train = "2_image\Train\$($d.Name)"
    ni -Path $path_test -ItemType Directory -Force
    ni -Path $path_train -ItemType Directory -Force    

    cp $test -destination $path_test        
    cp $train -destination $path_train
    
}
