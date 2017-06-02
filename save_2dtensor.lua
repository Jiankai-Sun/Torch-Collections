function save_2dtensor(tensor, filename)
  local out = assert(io.open(filename, 'w'))
  splitter = ","
  for i=1, tensor:size(1) do
    for j=1, tensor:size(2) do
      out:write(tensor[i][j])
      if j == tensor:size(2) then
        out:write("\n")
      else
        out:write(splitter)
      end
     end
   end
   out:close()
end
