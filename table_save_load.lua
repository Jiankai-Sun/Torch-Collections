--[[
   Save Table to File
   Load Table from File
   v 1.0
   
   Lua 5.2 compatible
   
   Only Saves Tables, Numbers and Strings
   Insides Table References are saved
   Does not save Userdata, Metatables, Functions and indices of these
   ----------------------------------------------------
   table.save( table , filename )
   
   on failure: returns an error msg
   
   ----------------------------------------------------
   table.load( filename or stringtable )
   
   Loads a table that has been saved via the table.save function
   
   on success: returns a previously saved table
   on failure: returns as second argument an error msg
   ----------------------------------------------------
   
   Licensed under the same terms as Lua itself.
]]--
do
   -- declare local variables
   --// exportstring( string )
   --// returns a "Lua" portable version of the string
   local function exportstring( s )
      return string.format("%q", s)
   end

   --// The Save Function
   function table.save(  tbl,filename )
      local charS,charE = "   ","\n"
      local file,err = io.open( filename, "wb" )
      if err then return err end

      -- initiate variables for save procedure
      local tables,lookup = { tbl },{ [tbl] = 1 }
      file:write( "return {"..charE )

      for idx,t in ipairs( tables ) do
         file:write( "-- Table: {"..idx.."}"..charE )
         file:write( "{"..charE )
         local thandled = {}

         for i,v in ipairs( t ) do
            thandled[i] = true
            local stype = type( v )
            -- only handle value
            if stype == "table" then
               if not lookup[v] then
                  table.insert( tables, v )
                  lookup[v] = #tables
               end
               file:write( charS.."{"..lookup[v].."},"..charE )
            elseif stype == "string" then
               file:write(  charS..exportstring( v )..","..charE )
            elseif stype == "number" then
               file:write(  charS..tostring( v )..","..charE )
            end
         end

         for i,v in pairs( t ) do
            -- escape handled values
            if (not thandled[i]) then
            
               local str = ""
               local stype = type( i )
               -- handle index
               if stype == "table" then
                  if not lookup[i] then
                     table.insert( tables,i )
                     lookup[i] = #tables
                  end
                  str = charS.."[{"..lookup[i].."}]="
               elseif stype == "string" then
                  str = charS.."["..exportstring( i ).."]="
               elseif stype == "number" then
                  str = charS.."["..tostring( i ).."]="
               end
            
               if str ~= "" then
                  stype = type( v )
                  -- handle value
                  if stype == "table" then
                     if not lookup[v] then
                        table.insert( tables,v )
                        lookup[v] = #tables
                     end
                     file:write( str.."{"..lookup[v].."},"..charE )
                  elseif stype == "string" then
                     file:write( str..exportstring( v )..","..charE )
                  elseif stype == "number" then
                     file:write( str..tostring( v )..","..charE )
                  end
               end
            end
         end
         file:write( "},"..charE )
      end
      file:write( "}" )
      file:close()
   end
   
   --// The Load Function
   function table.load( sfile )
      local ftables,err = loadfile( sfile )
      if err then return _,err end
      local tables = ftables()
      for idx = 1,#tables do
         local tolinki = {}
         for i,v in pairs( tables[idx] ) do
            if type( v ) == "table" then
               tables[idx][i] = tables[v[1]]
            end
            if type( i ) == "table" and tables[i[1]] then
               table.insert( tolinki,{ i,tables[i[1]] } )
            end
         end
         -- link indices
         for _,v in ipairs( tolinki ) do
            tables[idx][v[2]],tables[idx][v[1]] =  tables[idx][v[1]],nil
         end
      end
      return tables[1]
   end
-- close do
end

-- ChillCode
Testcode
dofile( "table.save-1.0.lua" )

function Main()
     
print( "Serialise Test ..." )
   
local t = {}
t.a = 1
t.b = 2
t.c = {}
-- self reference
t.c.a = t
t.inass = { 1,2,3,4,5,6,7,8,9,10 }
t.inasst = { {1},{2},{3},{4},{5},{6},{7},{8},{9},{10} }
-- random
t.f = { [{ a = 5, b = 7, }] = "helloooooooo", [{ 1,2,3, m = 5, 5,6,7 }] = "A Table", }

t.func = function(x,y)
   print( "Hello\nWorld" )
   local sum = x+y
   return sum
end

-- get test string, not string.char(26)
local str = ""
for i = 0, 255 do
   str = str..string.char( i )
end
t.lstri = { [str] = 1 }
t.lstrv = str

local function test() print("Hello") end

t[test] = 1

   
print( "\n## BEFORE SAVE ##" )

printtbl( t )

--// test save to file
assert( table.save( t, "test_tbl.lua" ) == nil )
   
-- load table from file
local t2,err = table.load( "test_tbl.lua" )

assert( err == nil )


print( "\n## AFTER SAVE ##" )

print( "\n## LOAD FROM FILE ##" )

printtbl( t2 )

print( "\n//Test References" )
   
assert( t2.c.a == t2 )
   
print( "\n//Test Long string" )

assert( t.lstrv == t2.lstrv )

print( "\n//Test Function\n\n" )

assert( t2.func == nil )

print( "\n*** Test SUCCESSFUL ***" )

end

function printtbl( t,tab,lookup )
   local lookup = lookup or { [t] = 1 }
   local tab = tab or ""
   for i,v in pairs( t ) do
      print( tab..tostring(i), v )
      if type(i) == "table" and not lookup[i] then
         lookup[i] = 1
         print( tab.."Table: i" )
         printtbl( i,tab.."\t",lookup )
      end
      if type(v) == "table" and not lookup[v] then
         lookup[v] = 1
         print( tab.."Table: v" )
         printtbl( v,tab.."\t",lookup )
      end
   end
end

function SaveHugeTable()

   local _t = {}

   for i=1,1000000 do

      local __t = {}
      table.insert(_t, __t)
      table.insert(__t, "THIS IS A STRING WITH A FEW CHARS")
      for i=1,30 do
         table.insert(__t, i)
      end   
   end
   
   print("Built Hunge Table!")

   t1 = os.clock()
   assert( table.save( _t, "test_huge_tbl.lua" ) == nil )
   print("Done Saving: "..os.difftime(os.clock(), t1))
   
   _t = nil
   collectgarbage()
   
   t1 = os.clock()
   local t2,err = table.load( "test_huge_tbl.lua" )
   print("Done Loading: "..os.difftime(os.clock(), t1))
   
   print("Num: "..#t2)

   assert( err == nil )
   
   print( "\n*** Test SUCCESSFUL ***" )
   
   io.read()
end


Main()
--SaveHugeTable()

io.read()
