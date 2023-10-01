(function() {var implementors = {
"flechasdb":[["impl Freeze for <a class=\"struct\" href=\"flechasdb/asyncdb/io/struct.LocalFileSystem.html\" title=\"struct flechasdb::asyncdb::io::LocalFileSystem\">LocalFileSystem</a>",1,["flechasdb::asyncdb::io::LocalFileSystem"]],["impl !Freeze for <a class=\"struct\" href=\"flechasdb/asyncdb/io/struct.LocalHashedFileIn.html\" title=\"struct flechasdb::asyncdb::io::LocalHashedFileIn\">LocalHashedFileIn</a>",1,["flechasdb::asyncdb::io::LocalHashedFileIn"]],["impl&lt;'db, 'i, 'k, T, FS, K: ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.72.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt; Freeze for <a class=\"struct\" href=\"flechasdb/asyncdb/stored/get_attribute/struct.GetAttributeInPartition.html\" title=\"struct flechasdb::asyncdb::stored::get_attribute::GetAttributeInPartition\">GetAttributeInPartition</a>&lt;'db, 'i, 'k, T, FS, K&gt;",1,["flechasdb::asyncdb::stored::get_attribute::GetAttributeInPartition"]],["impl&lt;'db, 'v, T, FS, V: ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.72.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>, EV&gt; Freeze for <a class=\"struct\" href=\"flechasdb/asyncdb/stored/query/struct.Query.html\" title=\"struct flechasdb::asyncdb::stored::query::Query\">Query</a>&lt;'db, 'v, T, FS, V, EV&gt;<span class=\"where fmt-newline\">where\n    EV: Freeze,</span>",1,["flechasdb::asyncdb::stored::query::Query"]],["impl&lt;'db, T, FS&gt; Freeze for <a class=\"struct\" href=\"flechasdb/asyncdb/stored/query/struct.QueryResult.html\" title=\"struct flechasdb::asyncdb::stored::query::QueryResult\">QueryResult</a>&lt;'db, T, FS&gt;<span class=\"where fmt-newline\">where\n    T: Freeze,</span>",1,["flechasdb::asyncdb::stored::query::QueryResult"]],["impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"flechasdb/asyncdb/stored/query/struct.PartitionQueryResult.html\" title=\"struct flechasdb::asyncdb::stored::query::PartitionQueryResult\">PartitionQueryResult</a>&lt;T&gt;<span class=\"where fmt-newline\">where\n    T: Freeze,</span>",1,["flechasdb::asyncdb::stored::query::PartitionQueryResult"]],["impl Freeze for <a class=\"enum\" href=\"flechasdb/asyncdb/stored/query/enum.QueryEvent.html\" title=\"enum flechasdb::asyncdb::stored::query::QueryEvent\">QueryEvent</a>",1,["flechasdb::asyncdb::stored::query::QueryEvent"]],["impl&lt;T, FS&gt; !Freeze for <a class=\"struct\" href=\"flechasdb/asyncdb/stored/struct.Database.html\" title=\"struct flechasdb::asyncdb::stored::Database\">Database</a>&lt;T, FS&gt;",1,["flechasdb::asyncdb::stored::Database"]],["impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"flechasdb/asyncdb/stored/struct.Partition.html\" title=\"struct flechasdb::asyncdb::stored::Partition\">Partition</a>&lt;T&gt;",1,["flechasdb::asyncdb::stored::Partition"]],["impl&lt;'a, T, VS&gt; Freeze for <a class=\"struct\" href=\"flechasdb/db/build/proto/struct.DatabaseSerialize.html\" title=\"struct flechasdb::db::build::proto::DatabaseSerialize\">DatabaseSerialize</a>&lt;'a, T, VS&gt;",1,["flechasdb::db::build::proto::DatabaseSerialize"]],["impl&lt;T, VS&gt; Freeze for <a class=\"struct\" href=\"flechasdb/db/build/struct.DatabaseBuilder.html\" title=\"struct flechasdb::db::build::DatabaseBuilder\">DatabaseBuilder</a>&lt;T, VS&gt;<span class=\"where fmt-newline\">where\n    VS: Freeze,</span>",1,["flechasdb::db::build::DatabaseBuilder"]],["impl&lt;'a, T&gt; Freeze for <a class=\"enum\" href=\"flechasdb/db/build/enum.BuildEvent.html\" title=\"enum flechasdb::db::build::BuildEvent\">BuildEvent</a>&lt;'a, T&gt;",1,["flechasdb::db::build::BuildEvent"]],["impl&lt;T, VS&gt; Freeze for <a class=\"struct\" href=\"flechasdb/db/build/struct.Database.html\" title=\"struct flechasdb::db::build::Database\">Database</a>&lt;T, VS&gt;<span class=\"where fmt-newline\">where\n    VS: Freeze,</span>",1,["flechasdb::db::build::Database"]],["impl&lt;'a, T, VS&gt; Freeze for <a class=\"struct\" href=\"flechasdb/db/build/struct.PartitionIter.html\" title=\"struct flechasdb::db::build::PartitionIter\">PartitionIter</a>&lt;'a, T, VS&gt;",1,["flechasdb::db::build::PartitionIter"]],["impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"flechasdb/db/build/struct.Partition.html\" title=\"struct flechasdb::db::build::Partition\">Partition</a>&lt;T&gt;",1,["flechasdb::db::build::Partition"]],["impl Freeze for <a class=\"enum\" href=\"flechasdb/db/build/enum.QueryEvent.html\" title=\"enum flechasdb::db::build::QueryEvent\">QueryEvent</a>",1,["flechasdb::db::build::QueryEvent"]],["impl&lt;'a, T, VS&gt; Freeze for <a class=\"struct\" href=\"flechasdb/db/build/struct.PartitionQuery.html\" title=\"struct flechasdb::db::build::PartitionQuery\">PartitionQuery</a>&lt;'a, T, VS&gt;",1,["flechasdb::db::build::PartitionQuery"]],["impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"flechasdb/db/build/struct.QueryResult.html\" title=\"struct flechasdb::db::build::QueryResult\">QueryResult</a>&lt;T&gt;<span class=\"where fmt-newline\">where\n    T: Freeze,</span>",1,["flechasdb::db::build::QueryResult"]],["impl&lt;T, FS&gt; !Freeze for <a class=\"struct\" href=\"flechasdb/db/stored/struct.Database.html\" title=\"struct flechasdb::db::stored::Database\">Database</a>&lt;T, FS&gt;",1,["flechasdb::db::stored::Database"]],["impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"flechasdb/db/stored/struct.Partition.html\" title=\"struct flechasdb::db::stored::Partition\">Partition</a>&lt;T&gt;",1,["flechasdb::db::stored::Partition"]],["impl Freeze for <a class=\"enum\" href=\"flechasdb/db/stored/enum.QueryEvent.html\" title=\"enum flechasdb::db::stored::QueryEvent\">QueryEvent</a>",1,["flechasdb::db::stored::QueryEvent"]],["impl&lt;'a, T, FS&gt; Freeze for <a class=\"struct\" href=\"flechasdb/db/stored/struct.QueryResult.html\" title=\"struct flechasdb::db::stored::QueryResult\">QueryResult</a>&lt;'a, T, FS&gt;<span class=\"where fmt-newline\">where\n    T: Freeze,</span>",1,["flechasdb::db::stored::QueryResult"]],["impl Freeze for <a class=\"enum\" href=\"flechasdb/db/enum.AttributeValue.html\" title=\"enum flechasdb::db::AttributeValue\">AttributeValue</a>",1,["flechasdb::db::AttributeValue"]],["impl&lt;X&gt; Freeze for <a class=\"struct\" href=\"flechasdb/distribution/struct.WeightedIndex.html\" title=\"struct flechasdb::distribution::WeightedIndex\">WeightedIndex</a>&lt;X&gt;<span class=\"where fmt-newline\">where\n    X: Freeze,\n    &lt;X as <a class=\"trait\" href=\"https://rust-random.github.io/rand/rand/distributions/uniform/trait.SampleUniform.html\" title=\"trait rand::distributions::uniform::SampleUniform\">SampleUniform</a>&gt;::<a class=\"associatedtype\" href=\"https://rust-random.github.io/rand/rand/distributions/uniform/trait.SampleUniform.html#associatedtype.Sampler\" title=\"type rand::distributions::uniform::SampleUniform::Sampler\">Sampler</a>: Freeze,</span>",1,["flechasdb::distribution::WeightedIndex"]],["impl Freeze for <a class=\"enum\" href=\"flechasdb/error/enum.Error.html\" title=\"enum flechasdb::error::Error\">Error</a>",1,["flechasdb::error::Error"]],["impl Freeze for <a class=\"struct\" href=\"flechasdb/io/struct.LocalFileSystem.html\" title=\"struct flechasdb::io::LocalFileSystem\">LocalFileSystem</a>",1,["flechasdb::io::LocalFileSystem"]],["impl Freeze for <a class=\"struct\" href=\"flechasdb/io/struct.LocalHashedFileOut.html\" title=\"struct flechasdb::io::LocalHashedFileOut\">LocalHashedFileOut</a>",1,["flechasdb::io::LocalHashedFileOut"]],["impl Freeze for <a class=\"struct\" href=\"flechasdb/io/struct.LocalHashedFileIn.html\" title=\"struct flechasdb::io::LocalHashedFileIn\">LocalHashedFileIn</a>",1,["flechasdb::io::LocalHashedFileIn"]],["impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"flechasdb/kmeans/struct.Codebook.html\" title=\"struct flechasdb::kmeans::Codebook\">Codebook</a>&lt;T&gt;",1,["flechasdb::kmeans::Codebook"]],["impl&lt;'a, T&gt; Freeze for <a class=\"enum\" href=\"flechasdb/kmeans/enum.ClusterEvent.html\" title=\"enum flechasdb::kmeans::ClusterEvent\">ClusterEvent</a>&lt;'a, T&gt;",1,["flechasdb::kmeans::ClusterEvent"]],["impl&lt;T, K, F&gt; Freeze for <a class=\"struct\" href=\"flechasdb/nbest/struct.NBestByKey.html\" title=\"struct flechasdb::nbest::NBestByKey\">NBestByKey</a>&lt;T, K, F&gt;<span class=\"where fmt-newline\">where\n    F: Freeze,</span>",1,["flechasdb::nbest::NBestByKey"]],["impl&lt;T, VS&gt; Freeze for <a class=\"struct\" href=\"flechasdb/partitions/struct.Partitions.html\" title=\"struct flechasdb::partitions::Partitions\">Partitions</a>&lt;T, VS&gt;<span class=\"where fmt-newline\">where\n    VS: Freeze,</span>",1,["flechasdb::partitions::Partitions"]],["impl&lt;'a, T, VS&gt; Freeze for <a class=\"struct\" href=\"flechasdb/partitions/struct.AllVectorIterator.html\" title=\"struct flechasdb::partitions::AllVectorIterator\">AllVectorIterator</a>&lt;'a, T, VS&gt;",1,["flechasdb::partitions::AllVectorIterator"]],["impl Freeze for <a class=\"enum\" href=\"flechasdb/protos/database/attribute_value/enum.Value.html\" title=\"enum flechasdb::protos::database::attribute_value::Value\">Value</a>",1,["flechasdb::protos::database::attribute_value::Value"]],["impl !Freeze for <a class=\"struct\" href=\"flechasdb/protos/database/struct.Database.html\" title=\"struct flechasdb::protos::database::Database\">Database</a>",1,["flechasdb::protos::database::Database"]],["impl !Freeze for <a class=\"struct\" href=\"flechasdb/protos/database/struct.Partition.html\" title=\"struct flechasdb::protos::database::Partition\">Partition</a>",1,["flechasdb::protos::database::Partition"]],["impl !Freeze for <a class=\"struct\" href=\"flechasdb/protos/database/struct.VectorSet.html\" title=\"struct flechasdb::protos::database::VectorSet\">VectorSet</a>",1,["flechasdb::protos::database::VectorSet"]],["impl !Freeze for <a class=\"struct\" href=\"flechasdb/protos/database/struct.EncodedVectorSet.html\" title=\"struct flechasdb::protos::database::EncodedVectorSet\">EncodedVectorSet</a>",1,["flechasdb::protos::database::EncodedVectorSet"]],["impl !Freeze for <a class=\"struct\" href=\"flechasdb/protos/database/struct.AttributeValue.html\" title=\"struct flechasdb::protos::database::AttributeValue\">AttributeValue</a>",1,["flechasdb::protos::database::AttributeValue"]],["impl !Freeze for <a class=\"struct\" href=\"flechasdb/protos/database/struct.AttributesLog.html\" title=\"struct flechasdb::protos::database::AttributesLog\">AttributesLog</a>",1,["flechasdb::protos::database::AttributesLog"]],["impl !Freeze for <a class=\"struct\" href=\"flechasdb/protos/database/struct.OperationSetAttribute.html\" title=\"struct flechasdb::protos::database::OperationSetAttribute\">OperationSetAttribute</a>",1,["flechasdb::protos::database::OperationSetAttribute"]],["impl !Freeze for <a class=\"struct\" href=\"flechasdb/protos/database/struct.Uuid.html\" title=\"struct flechasdb::protos::database::Uuid\">Uuid</a>",1,["flechasdb::protos::database::Uuid"]],["impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"flechasdb/vector/struct.BlockVectorSet.html\" title=\"struct flechasdb::vector::BlockVectorSet\">BlockVectorSet</a>&lt;T&gt;",1,["flechasdb::vector::BlockVectorSet"]],["impl&lt;'a, T, VS&gt; Freeze for <a class=\"struct\" href=\"flechasdb/vector/struct.SubVectorSet.html\" title=\"struct flechasdb::vector::SubVectorSet\">SubVectorSet</a>&lt;'a, T, VS&gt;",1,["flechasdb::vector::SubVectorSet"]]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()